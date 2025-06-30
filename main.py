from abc import ABC, abstractmethod
from manim import *
import numpy as np
import random

# === CONFIGURATION ===
config = {
    "scene": {
        "wait_time": 6,
        # vertical separation between interactive walls
        "wall_spacing": 1.0,
        # small gap between interactive and non-interactable walls
        "non_wall_gap": 0.1
    },
    "wall": {
        "start": LEFT * 3.5,
        "end": RIGHT * 3.5,
        "color": BLUE,
        "max_acceptance_angle_deg": 30,
        "refractive_index_up": 1.0,
        "refractive_index_down": 1.333,
        "immutable_steps": 10,
    },
    "non_interactable_wall": {
        # faded blue
        "color": BLUE,
        "opacity": 0.5,
        "stroke_width": 4,
    },
    "cavity": {
        "radius": 0.05,
        "color": RED,
        # maximum vertical jitter within walls
        "vertical_jitter": 0.2,
    },
    "source": {
        "type": "screen",
        "mode": "ray",
        # just smaller than wall_spacing (1.0 → 0.9 total length)
        "start": LEFT * 4 + DOWN * 0.4,
        "end": LEFT * 4 + UP * 0.4,
        "tilt_angle_deg": 180,
        "emission_cone_deg": 100,
        "count": 20,
        "distance": 0.02,
        "ray_length": 1,
        "speed": 3,
        "radius": 0.03,
        "color": GREEN,
    },
    "receiver": {
        # slightly shorter than wall_spacing
        "length": 0.9,
        # horizontal margin to the right
        "position": RIGHT * 4,
        "color": YELLOW,
        "stroke_width": 4,
        "tilt_angle_deg": 90,
    },
}



# === NON-INTERACTABLE CLASSES ===
class NonInteractable:
    def __init__(self, mob):
        self.mob = mob
    def add_to_scene(self, scene: Scene):
        scene.add(self.mob)

class NonInteractableWall(NonInteractable):
    def __init__(self, start, end, color=GRAY, **kwargs):
        line = Line(start, end, color=color, **kwargs)
        super().__init__(line)

# === BASE INTERFACE ===
class Interactable(ABC):
    @abstractmethod
    def check_collision(self, particle) -> bool: ...
    @abstractmethod
    def interact(self, particle): ...
    @abstractmethod
    def add_to_scene(self, scene: Scene): ...

# === ABSORBING RECEIVER ===
class Receiver(Interactable):
    def __init__(self, position, length, tilt_angle_deg, color, stroke_width):
        half = length / 2
        start = np.array(position) + np.array([-half, 0, 0])
        end   = np.array(position) + np.array([ half, 0, 0])
        line = Line(start, end, color=color, stroke_width=stroke_width)
        center = np.array(position)
        line.rotate(np.deg2rad(tilt_angle_deg), about_point=center)
        self.line = line

        self.start = np.array(line.get_start())
        self.end   = np.array(line.get_end())
        v = self.end - self.start
        n = np.array([-v[1], v[0], 0])
        self.normal = n / np.linalg.norm(n)
        self.point_on_line = self.start

        # placeholder for scene reference
        self._scene = None

    def check_collision(self, particle) -> bool:
        p = particle.mob.get_center()
        v_line = self.end - self.start
        L = np.linalg.norm(v_line)
        v_to_p = p - self.point_on_line
        proj = np.dot(v_to_p, v_line) / L
        if proj < 0 or proj > L:
            return False
        dist = abs(np.dot(v_to_p, self.normal))
        return dist <= particle.radius

    def interact(self, particle):
        # stop motion
        particle.mob.clear_updaters()
        # make particle transparent by setting opacity to 0
        particle.mob.set_opacity(0)

    def add_to_scene(self, scene: Scene):
        # remember the scene so we can remove things later
        self._scene = scene
        scene.add(self.line)



# === WALL ===
class Wall(Interactable):
    def __init__(
        self, start, end, color,
        max_acceptance_angle_deg,
        refractive_index_up,
        refractive_index_down,
        immutable_steps,
        **kwargs
    ):
        self.line = Line(start, end, color=color, **kwargs)
        v = np.array(end) - np.array(start)
        n = np.array([-v[1], v[0], 0])
        self.normal = n / np.linalg.norm(n)
        self.point_on_line = np.array(start)
        self.max_acceptance = np.deg2rad(max_acceptance_angle_deg)
        self.n_up = refractive_index_up
        self.n_down = refractive_index_down
        self.immutable_steps = immutable_steps

    def check_collision(self, particle) -> bool:
        p = particle.mob.get_center()
        v_line = np.array(self.line.get_end()) - np.array(self.line.get_start())
        L = np.linalg.norm(v_line)
        v_to_p = p - self.point_on_line
        proj = np.dot(v_to_p, v_line) / L
        if proj < 0 or proj > L:
            return False
        dist = abs(np.dot(v_to_p, self.normal))
        return dist <= particle.radius

    def interact(self, particle):
        u = particle.velocity
        speed = np.linalg.norm(u)
        
        # Skip interaction if particle has no velocity
        if speed == 0:
            return
            
        û = u / speed
        
        # Find the correct face normal (the one the particle is approaching)
        face_n = None
        for n in (self.normal, -self.normal):
            if np.dot(û, n) < 0:
                face_n = n
                break
        
        # If no valid face normal found, default to reflection
        if face_n is None:
            face_n = self.normal
            
        cos_i = -np.dot(û, face_n)
        θi = np.arccos(np.clip(cos_i, -1, 1))
        
        if np.allclose(face_n, self.normal):
            n1, n2 = self.n_up, self.n_down
        else:
            n1, n2 = self.n_down, self.n_up
            
        if θi < self.max_acceptance:
            ratio = n1 / n2
            sin_t = ratio * np.sin(θi)
            if abs(sin_t) > 1:
                # Total internal reflection
                v_new = u - 2 * np.dot(u, face_n) * face_n
            else:
                # Refraction
                θt = np.arcsin(sin_t)
                t = û - (û.dot(face_n)) * face_n
                t_norm = np.linalg.norm(t)
                if t_norm > 0:
                    t_hat = t / t_norm
                    v_hat_new = -np.cos(θt) * face_n + np.sin(θt) * t_hat
                    v_new = speed * v_hat_new
                else:
                    # Handle edge case where tangential component is zero
                    v_new = u - 2 * np.dot(u, face_n) * face_n
            particle.immutable_counter = self.immutable_steps
        else:
            # Reflection
            v_new = u - 2 * np.dot(u, face_n) * face_n
            
        particle.velocity = v_new
        vx, vy = v_new[:2]
        particle.angle = np.degrees(np.arctan2(vy, vx))

    def add_to_scene(self, scene: Scene):
        scene.add(self.line)

# === CAVITY ===
class Cavity(Interactable):
    def __init__(self, shape: Mobject):
        self.shape = shape
    def check_collision(self, particle) -> bool:
        p_pos = particle.mob.get_center()
        center = self.shape.get_center()
        dist = np.linalg.norm(p_pos - center)
        r_shape = getattr(self.shape, "radius", 0)
        return dist <= (r_shape + particle.radius)
    def interact(self, particle):
        center = self.shape.get_center()
        particle.mob.move_to(center)
        
        # Preserve the current speed magnitude
        current_speed = np.linalg.norm(particle.velocity)
        
        angle_deg = random.uniform(0, 360)
        particle.angle = angle_deg
        θ = np.deg2rad(angle_deg)
        particle.velocity = current_speed * np.array([np.cos(θ), np.sin(θ), 0])
        r = getattr(self.shape, "radius", 0)
        particle.immutable_counter = int((2 * r) / current_speed * 60)
    def add_to_scene(self, scene: Scene):
        scene.add(self.shape)

# === PARTICLE ===
class Particle:
    def __init__(self, position, angle_deg, speed, radius, color, interactables):
        self.speed = speed
        θ = np.deg2rad(angle_deg)
        self.velocity = speed * np.array([np.cos(θ), np.sin(θ), 0])
        self.angle = angle_deg
        self.radius = radius
        self.interactables = interactables
        self.immutable_counter = 0
        self.mob = Dot(point=position, radius=radius, color=color)
        self.mob.add_updater(self._update)
    def _update(self, mobj, dt):
        mobj.shift(self.velocity * dt)
        if self.immutable_counter > 0:
            self.immutable_counter -= 1
            return
        for obj in self.interactables:
            if obj.check_collision(self):
                obj.interact(self)
    def add_to_scene(self, scene):
        scene.add(self.mob)

# === PARTICLE GENERATOR ===
class ParticleGenerator(ABC):
    @abstractmethod
    def generate(self, scene): ...
    @abstractmethod
    def add_to_scene(self, scene: Scene): ...

# === SCREEN SOURCE ===
class ScreenSource(ParticleGenerator):
    def __init__(
        self, start, end,
        tilt_angle_deg, emission_cone_deg,
        count, distance, ray_length, mode,
        speed, radius, color,
        interactables
    ):
        self.start = np.array(start)
        self.end = np.array(end)
        self.tilt_angle = np.deg2rad(tilt_angle_deg)
        self.cone = emission_cone_deg
        self.count = count
        self.distance = distance
        self.ray_length = ray_length
        self.mode = mode
        self.speed = speed
        self.radius = radius
        self.color = color
        self.interactables = interactables
        self._ray_particles = []
        self._timer = 0.0
        self._interval = distance / speed
        self._wave_idx = 0
        self.line_mob = None
        
    def add_to_scene(self, scene: Scene):
        line = Line(self.start, self.end, color=self.color)
        center = (self.start + self.end) / 2
        line.rotate(self.tilt_angle, about_point=center)
        scene.add(line)
        self.line_mob = line
        self.start_world = line.get_start()
        self.end_world = line.get_end()

    def generate(self, scene):
        v = self.end_world - self.start_world
        normal = np.array([-v[1], v[0], 0])
        normal /= np.linalg.norm(normal)
        base_angle = np.degrees(np.arctan2(normal[1], normal[0]))
        origins, directions = [], []
        for _ in range(self.count):
            t = random.random()
            origin = self.start_world + t * (self.end_world - self.start_world)
            ray_ang = base_angle + random.uniform(-self.cone/2, self.cone/2)
            θ = np.deg2rad(ray_ang)
            dir_v = np.array([np.cos(θ), np.sin(θ), 0])
            origins.append(origin)
            directions.append(dir_v)
        steps = int(self.ray_length / self.distance)
        for ori, dir_v in zip(origins, directions):
            ray_list = []
            for _ in range(steps + 1):
                p = Particle(
                    position=ori,
                    angle_deg=0,
                    speed=0,
                    radius=self.radius,
                    color=self.color,
                    interactables=self.interactables
                )
                p._dir = dir_v
                p.add_to_scene(scene)
                ray_list.append(p)
            self._ray_particles.append(ray_list)
        self.line_mob.add_updater(self._release_next)

    def _release_next(self, mobj, dt):
        self._timer += dt
        while self._timer >= self._interval:
            self._timer -= self._interval
            for ray in self._ray_particles:
                if self._wave_idx < len(ray):
                    p = ray[self._wave_idx]
                    # don’t touch p.speed—just set its velocity from the fixed speed
                    p.velocity = p._dir * self.speed
                    vx, vy = p.velocity[:2]
                    p.angle = np.degrees(np.arctan2(vy, vx))
            self._wave_idx += 1
        if self._wave_idx >= len(self._ray_particles[0]):
            mobj.remove_updater(self._release_next)

# === SCENE ===
class GeneralSimulation(Scene):
    def construct(self):
        w = config["wall"]
        spacing = config["scene"]["wall_spacing"]
        gap = config["scene"]["non_wall_gap"]

        # Interactive walls (top & bottom)
        top_offset = UP * (spacing / 2)
        bottom_offset = DOWN * (spacing / 2)
        wall_top = Wall(
            start=w["start"] + top_offset,
            end=w["end"] + top_offset,
            color=w["color"],
            max_acceptance_angle_deg=w["max_acceptance_angle_deg"],
            refractive_index_up=w["refractive_index_up"],
            refractive_index_down=w["refractive_index_down"],
            immutable_steps=w["immutable_steps"],
        )
        wall_top.add_to_scene(self)
        wall_bottom = Wall(
            start=w["start"] + bottom_offset,
            end=w["end"] + bottom_offset,
            color=w["color"],
            max_acceptance_angle_deg=w["max_acceptance_angle_deg"],
            refractive_index_up=w["refractive_index_up"],
            refractive_index_down=w["refractive_index_down"],
            immutable_steps=w["immutable_steps"],
        )
        wall_bottom.add_to_scene(self)

        # Non-interactable horizontal walls (just outside)
        niw_conf = config["non_interactable_wall"]
        non_top = NonInteractableWall(
            start=w["start"] + UP * (spacing / 2 + gap),
            end=w["end"] + UP * (spacing / 2 + gap),
            color=niw_conf["color"], stroke_width=niw_conf["stroke_width"]
        )
        non_top.add_to_scene(self)
        non_top.mob.set_opacity(niw_conf["opacity"])
        non_bottom = NonInteractableWall(
            start=w["start"] + DOWN * (spacing / 2 + gap),
            end=w["end"] + DOWN * (spacing / 2 + gap),
            color=niw_conf["color"], stroke_width=niw_conf["stroke_width"]
        )
        non_bottom.add_to_scene(self)
        non_bottom.mob.set_opacity(niw_conf["opacity"])

        # Non-interactable vertical walls (connect horizontals)
        left_x = w["start"][0]
        right_x = w["end"][0]
        y_top = spacing / 2 + gap
        y_bottom = -spacing / 2 - gap
        left_vert = NonInteractableWall(
            start=LEFT * abs(left_x) + DOWN * y_top * 0 + UP * 0,  # placeholder
            end=LEFT * abs(left_x) + DOWN * y_top * 0 + UP * 0,    # placeholder
            color=niw_conf["color"], stroke_width=niw_conf["stroke_width"]
        )
        # Actually define using points:
        left_vert = NonInteractableWall(
            start=LEFT * abs(left_x) + DOWN * (spacing / 2 + gap),
            end=LEFT * abs(left_x) + UP * (spacing / 2 + gap),
            color=niw_conf["color"], stroke_width=niw_conf["stroke_width"]
        )
        left_vert.add_to_scene(self)
        left_vert.mob.set_opacity(niw_conf["opacity"])
        right_vert = NonInteractableWall(
            start=RIGHT * abs(right_x) + DOWN * (spacing / 2 + gap),
            end=RIGHT * abs(right_x) + UP * (spacing / 2 + gap),
            color=niw_conf["color"], stroke_width=niw_conf["stroke_width"]
        )
        right_vert.add_to_scene(self)
        right_vert.mob.set_opacity(niw_conf["opacity"])

        # Cavities (three, varying radii and slight vertical variation)
        c_conf = config["cavity"]
        radii = [c_conf["radius"] * 0.8, c_conf["radius"], c_conf["radius"] * 1.2]
        xs = np.linspace(w["start"][0] + 1, w["end"][0] - 1, len(radii))
        cavities = []
        for r, x in zip(radii, xs):
            dy = random.uniform(-c_conf["vertical_jitter"], c_conf["vertical_jitter"])
            # ensure cavities remain within interactive walls
            y = np.clip(dy, -spacing/2 + r, spacing/2 - r)
            cav_shape = Circle(radius=r, color=c_conf["color"]) .move_to([x, y, 0])
            cav = Cavity(cav_shape)
            cav.add_to_scene(self)
            cavities.append(cav)

        # Receiver at right
        r_conf = config["receiver"]
        receiver = Receiver(
            position=r_conf["position"],
            length=r_conf["length"],
            tilt_angle_deg=r_conf["tilt_angle_deg"],
            color=r_conf["color"], stroke_width=r_conf["stroke_width"]
        )
        receiver.add_to_scene(self)

        # Source on the left
        s = config["source"]
        generator = ScreenSource(
            start=s["start"], end=s["end"], mode=s["mode"], tilt_angle_deg=s["tilt_angle_deg"],
            emission_cone_deg=s["emission_cone_deg"], count=s["count"],
            distance=s["distance"], ray_length=s["ray_length"], speed=s["speed"],
            radius=s["radius"], color=s["color"], interactables=[wall_top, wall_bottom] + cavities + [receiver]
        )
        generator.add_to_scene(self)
        generator.generate(self)

        self.wait(config["scene"]["wait_time"])