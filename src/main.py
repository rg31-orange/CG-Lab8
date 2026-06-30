import taichi as ti


# ============================================================
# Taichi Initialization
# ============================================================

ti.init(arch=ti.gpu)


# ============================================================
# Global Parameters
# ============================================================

N = 24
NUM_PARTICLES = N * N

mass = 1.0
dt = 5e-4

gravity = ti.Vector([0.0, -9.8, 0.0])

# Spring parameters
k_s = ti.field(dtype=ti.f32, shape=())
k_d = ti.field(dtype=ti.f32, shape=())
max_velocity = ti.field(dtype=ti.f32, shape=())

# Simulation switches
enable_shear = ti.field(dtype=ti.i32, shape=())
enable_bending = ti.field(dtype=ti.i32, shape=())
enable_collision = ti.field(dtype=ti.i32, shape=())

# Collision sphere
# scene.particles() requires a 1D field, so shape=1.
sphere_center = ti.Vector.field(3, dtype=ti.f32, shape=1)
sphere_radius = ti.field(dtype=ti.f32, shape=())

# Cloth fields
x = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
v = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
f = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
is_fixed = ti.field(dtype=ti.i32, shape=NUM_PARTICLES)

# Implicit Euler temporary fields
x_next = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
v_next = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
f_next = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)

# Spring data
MAX_SPRINGS = N * N * 8

spring_pairs = ti.Vector.field(2, dtype=ti.i32, shape=MAX_SPRINGS)
spring_lengths = ti.field(dtype=ti.f32, shape=MAX_SPRINGS)
spring_type = ti.field(dtype=ti.i32, shape=MAX_SPRINGS)
num_springs = ti.field(dtype=ti.i32, shape=())

# Rendering indices for scene.lines
spring_indices = ti.field(dtype=ti.i32, shape=MAX_SPRINGS * 2)


# ============================================================
# Initialization Kernels
# ============================================================

@ti.kernel
def init_global_params():
    k_s[None] = 8000.0
    k_d[None] = 1.0
    max_velocity[None] = 50.0

    enable_shear[None] = 1
    enable_bending[None] = 1
    enable_collision[None] = 1

    # Lower and slightly shrink the sphere.
    # This makes the cloth visually fall onto the sphere instead of being hidden by it.
    sphere_center[0] = ti.Vector([0.0, -0.08, 0.0])
    sphere_radius[None] = 0.20


@ti.kernel
def init_positions():
    """
    Initialize particle positions and fixed constraints.
    The cloth is placed in the XZ plane and falls along Y axis.
    """
    for i, j in ti.ndrange(N, N):
        idx = i * N + j

        spacing = 1.0 / (N - 1)
        px = i * spacing - 0.5
        py = 0.75
        pz = j * spacing - 0.5

        x[idx] = ti.Vector([px, py, pz])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])

        # Fix two top corners.
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0


@ti.func
def add_spring(a: ti.i32, b: ti.i32, t: ti.i32):
    c = ti.atomic_add(num_springs[None], 1)

    if c < MAX_SPRINGS:
        spring_pairs[c] = ti.Vector([a, b])
        spring_lengths[c] = (x[a] - x[b]).norm()
        spring_type[c] = t


@ti.kernel
def init_structural_springs():
    """
    Structural springs:
    connect horizontal and vertical neighbors.
    spring_type = 0
    """
    for i, j in ti.ndrange(N, N):
        idx = i * N + j

        if i < N - 1:
            idx_right = (i + 1) * N + j
            add_spring(idx, idx_right, 0)

        if j < N - 1:
            idx_down = i * N + (j + 1)
            add_spring(idx, idx_down, 0)


@ti.kernel
def init_shear_springs():
    """
    Shear springs:
    connect diagonal neighbors.
    spring_type = 1
    """
    for i, j in ti.ndrange(N - 1, N - 1):
        idx = i * N + j
        idx_right_down = (i + 1) * N + (j + 1)

        idx_right = (i + 1) * N + j
        idx_down = i * N + (j + 1)

        add_spring(idx, idx_right_down, 1)
        add_spring(idx_right, idx_down, 1)


@ti.kernel
def init_bending_springs():
    """
    Bending springs:
    connect particles two grid units apart.
    spring_type = 2
    """
    for i, j in ti.ndrange(N, N):
        idx = i * N + j

        if i < N - 2:
            idx_far_x = (i + 2) * N + j
            add_spring(idx, idx_far_x, 2)

        if j < N - 2:
            idx_far_z = i * N + (j + 2)
            add_spring(idx, idx_far_z, 2)


@ti.kernel
def init_spring_indices():
    """
    Prepare index buffer for GGUI line rendering.
    """
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]


def init_cloth():
    """
    Python side sequential kernel calls guarantee GPU synchronization.
    """
    num_springs[None] = 0

    init_positions()
    init_structural_springs()

    if enable_shear[None] == 1:
        init_shear_springs()

    if enable_bending[None] == 1:
        init_bending_springs()

    init_spring_indices()


# ============================================================
# Physics Functions
# ============================================================

@ti.func
def get_spring_stiffness(t: ti.i32) -> ti.f32:
    """
    Use different stiffness for different spring types.
    """
    stiffness = k_s[None]

    if t == 1:
        stiffness = k_s[None] * 0.75
    elif t == 2:
        stiffness = k_s[None] * 0.35

    return stiffness


@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    """
    Compute gravity, damping and spring forces.
    This function is inlined into kernels by ti.func.
    """
    # Clear force and apply gravity + damping.
    for i in range(NUM_PARTICLES):
        force[i] = gravity * mass - k_d[None] * vel[i]

    # Accumulate spring forces.
    for s in range(num_springs[None]):
        idx_a = spring_pairs[s][0]
        idx_b = spring_pairs[s][1]

        pos_a = pos[idx_a]
        pos_b = pos[idx_b]

        d = pos_a - pos_b
        dist = d.norm()

        if dist > 1e-6:
            direction = d / dist
            stiffness = get_spring_stiffness(spring_type[s])
            rest_len = spring_lengths[s]

            f_spring = -stiffness * (dist - rest_len) * direction

            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)


@ti.func
def clamp_velocity(vel: ti.template(), idx: ti.i32):
    """
    Clamp velocity to avoid numerical explosion.
    """
    speed = vel[idx].norm()

    if speed > max_velocity[None]:
        vel[idx] = vel[idx] / speed * max_velocity[None]


@ti.func
def solve_sphere_collision(pos: ti.template(), vel: ti.template(), idx: ti.i32):
    """
    Simple particle-sphere collision.
    If a particle enters the sphere, project it to the surface
    and remove inward velocity component.
    """
    if enable_collision[None] == 1 and is_fixed[idx] == 0:
        center = sphere_center[0]
        r = sphere_radius[None]

        d = pos[idx] - center
        dist = d.norm()

        if dist < r and dist > 1e-6:
            n = d / dist

            # Position projection.
            pos[idx] = center + n * r

            # Remove inward velocity.
            vn = vel[idx].dot(n)
            if vn < 0.0:
                vel[idx] = vel[idx] - vn * n

            # Slight tangential damping.
            vel[idx] *= 0.98


# ============================================================
# Integration Kernels
# ============================================================

@ti.kernel
def step_explicit():
    """
    Explicit Euler:
    x_{t+1} = x_t + v_t dt
    v_{t+1} = v_t + a_t dt
    """
    compute_forces_on(x, v, f)

    for i in range(NUM_PARTICLES):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt

            clamp_velocity(v, i)
            solve_sphere_collision(x, v, i)


@ti.kernel
def step_semi_implicit():
    """
    Semi-Implicit Euler:
    v_{t+1} = v_t + a_t dt
    x_{t+1} = x_t + v_{t+1} dt
    """
    compute_forces_on(x, v, f)

    for i in range(NUM_PARTICLES):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)

            x[i] += v[i] * dt
            solve_sphere_collision(x, v, i)


@ti.kernel
def step_implicit_iter():
    """
    Implicit Euler approximated by fixed-point iterations:
    v_{t+1} = v_t + a(x_{t+1}, v_{t+1}) dt
    x_{t+1} = x_t + v_{t+1} dt
    """
    # Copy current state to predicted state.
    for i in range(NUM_PARTICLES):
        x_next[i] = x[i]
        v_next[i] = v[i]

    # Fixed-point iterations.
    for _ in ti.static(range(5)):
        compute_forces_on(x_next, v_next, f_next)

        for i in range(NUM_PARTICLES):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)

                x_next[i] = x[i] + v_next[i] * dt
                solve_sphere_collision(x_next, v_next, i)

    # Write back.
    for i in range(NUM_PARTICLES):
        if is_fixed[i] == 0:
            x[i] = x_next[i]
            v[i] = v_next[i]
        else:
            v[i] = ti.Vector([0.0, 0.0, 0.0])


# ============================================================
# GUI Helper
# ============================================================

def draw_gui(window, state):
    """
    Python-side GUI panel.
    """
    window.GUI.begin("Control Panel", 0.02, 0.02, 0.42, 0.56)

    window.GUI.text("Mass-Spring Cloth Simulation")
    window.GUI.text("")

    window.GUI.text("Integration Method:")

    prefix_0 = "[*] " if state["method"] == 0 else "[ ] "
    prefix_1 = "[*] " if state["method"] == 1 else "[ ] "
    prefix_2 = "[*] " if state["method"] == 2 else "[ ] "

    if window.GUI.button(prefix_0 + "Explicit Euler"):
        state["method"] = 0
        init_cloth()

    if window.GUI.button(prefix_1 + "Semi-Implicit Euler"):
        state["method"] = 1
        init_cloth()

    if window.GUI.button(prefix_2 + "Implicit Euler"):
        state["method"] = 2
        init_cloth()

    window.GUI.text("")

    pause_label = "Resume Simulation" if state["paused"] else "Pause Simulation"
    if window.GUI.button(pause_label):
        state["paused"] = not state["paused"]

    if window.GUI.button("Reset Cloth"):
        init_cloth()

    window.GUI.text("")

    if window.GUI.button("Toggle Shear Springs"):
        enable_shear[None] = 1 - enable_shear[None]
        init_cloth()

    if window.GUI.button("Toggle Bending Springs"):
        enable_bending[None] = 1 - enable_bending[None]
        init_cloth()

    if window.GUI.button("Toggle Sphere Collision"):
        enable_collision[None] = 1 - enable_collision[None]

    window.GUI.text("")

    # GUI sliders.
    damping = window.GUI.slider_float("Damping", k_d[None], 0.0, 10.0)
    stiffness = window.GUI.slider_float("Stiffness", k_s[None], 1000.0, 20000.0)
    max_v = window.GUI.slider_float("Max Velocity", max_velocity[None], 5.0, 100.0)

    k_d[None] = damping
    k_s[None] = stiffness
    max_velocity[None] = max_v

    window.GUI.text("")
    window.GUI.text(f"Shear Springs: {'ON' if enable_shear[None] == 1 else 'OFF'}")
    window.GUI.text(f"Bending Springs: {'ON' if enable_bending[None] == 1 else 'OFF'}")
    window.GUI.text(f"Sphere Collision: {'ON' if enable_collision[None] == 1 else 'OFF'}")
    window.GUI.text(f"Spring Count: {num_springs[None]}")

    window.GUI.end()


# ============================================================
# Main Application
# ============================================================

def main():
    init_global_params()
    init_cloth()

    window = ti.ui.Window(
        "CG Lab 7 - Mass Spring Cloth Model",
        (1024, 768),
        vsync=True,
    )

    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    camera.position(0.0, 0.35, 2.2)
    camera.lookat(0.0, 0.15, 0.0)

    state = {
        "method": 1,   # 0 explicit, 1 semi-implicit, 2 implicit
        "paused": False,
    }

    while window.running:
        draw_gui(window, state)

        if not state["paused"]:
            # More substeps improve stability.
            for _ in range(35):
                if state["method"] == 0:
                    step_explicit()
                elif state["method"] == 1:
                    step_semi_implicit()
                else:
                    step_implicit_iter()

        camera.track_user_inputs(
            window,
            movement_speed=0.03,
            hold_key=ti.ui.RMB,
        )

        scene.set_camera(camera)
        scene.ambient_light((0.45, 0.45, 0.45))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1.0, 1.0, 1.0))
        scene.point_light(pos=(-0.8, 1.2, 0.8), color=(0.6, 0.6, 0.7))

        # Draw collision sphere.
        # The color is darker to avoid visually covering the blue cloth too much.
        if enable_collision[None] == 1:
            scene.particles(
                sphere_center,
                radius=sphere_radius[None],
                color=(0.75, 0.35, 0.20),
            )

        # Draw cloth particles.
        # Larger blue particles make the cloth visually clearer in GIF.
        scene.particles(
            x,
            radius=0.014,
            color=(0.05, 0.45, 1.0),
        )

        # Draw spring lines.
        # Brighter and thicker lines make the cloth grid easier to observe.
        scene.lines(
            x,
            indices=spring_indices,
            width=1.8,
            color=(0.95, 0.95, 0.95),
        )

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
