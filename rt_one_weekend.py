import math
import random
import time
from multiprocessing import Pool


class Vec3:
    def __init__(self, x=0, y=0, z=0):
        self._x = x
        self._y = y
        self._z = z

    def x(self):
        return self._x

    def y(self):
        return self._y

    def z(self):
        return self._z

    def __neg__(self):
        return Vec3(-self._x, -self._y, -self._z)

    def __add__(self, other):
        return Vec3(self.x() + other.x(), self.y() + other.y(), self.z() + other.z())

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return Vec3(self.x() - other.x(), self.y() - other.y(), self.z() - other.z())

    def __iadd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if type(other) is Vec3:
            return Vec3(self._x * other.x(), self._y * other.y(), self._z * other.z())
        elif type(other) is int or type(other) is float:
            return Vec3(self._x * other, self._y * other, self._z * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Vec3(self._x / other, self._y / other, self._z / other)

    def __repr__(self):
        return "(" + str(self._x) + ", " + str(self._y) + ", " + str(self._z) + ")"

    def length_squared(self):
        return self._x**2 + self._y**2 + self._z**2

    def length(self):
        return math.sqrt(self.length_squared())

    def set(self, v):
        self._x = v.x()
        self._y = v.y()
        self._z = v.z()

    @staticmethod
    def dot(u, v):
        return u.x() * v.x() + u.y() * v.y() + u.z() * v.z()

    @staticmethod
    def cross(u, v):
        return Vec3(u.y() * v.z() - u.z() * v.y(), u.z() * v.x() - u.x() * v.z(), u.x() * v.y() - u.y() * v.x())

    @staticmethod
    def unit_vector(u):
        return u / u.length()

    @staticmethod
    def random(min=0.0, max=1.0):
        return Vec3(random_double(min, max), random_double(min, max), random_double(min, max))

# Vec3 aliaces
Color = Vec3
Point3 = Vec3


class Ray:
    def __init__(self, origin=None, direction=None):
        self._origin = origin
        self._direction = direction

    def origin(self):
        return self._origin

    def direction(self):
        return self._direction

    def at(self, t):
        return self._origin + self._direction * t

    def set(self, ray):
        self._direction = ray.direction()
        self._origin = ray.origin()


class HitRecord:
    def __init__(self):
        self.p = None
        self.normal = None
        self.t = None
        self.material = None
        self.front_face = None

    def set_face_normal(self, ray, outward_normal):
        self.front_face = Vec3.dot(ray.direction(), outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal


class Sphere:
    def __init__(self, center, radius, material):
        self._center = center
        self._radius = radius
        self._material = material

    def hit(self, ray, t_min, t_max, rec):
        oc = ray.origin() - self._center
        a = ray.direction().length_squared()
        half_b = Vec3.dot(oc, ray.direction())
        c = oc.length_squared() - self._radius**2
        discriminant = half_b * half_b - a * c
        if discriminant > 0:
            root = math.sqrt(discriminant)
            temp = (-half_b - root)/a
            if temp < t_max and temp > t_min:
                rec.t = temp
                rec.p = ray.at(rec.t)
                outward_normal = (rec.p - self._center) / self._radius
                rec.set_face_normal(ray, outward_normal)
                rec.material = self._material
                return True
            temp = (-half_b + root) / a
            if temp < t_max and temp > t_min:
                rec.t = temp
                rec.p = ray.at(rec.t)
                outward_normal = (rec.p - self._center) / self._radius
                rec.set_face_normal(ray, outward_normal)
                rec.material = self._material
                return True
        return False


class Lambertian:
    def __init__(self, color):
        self._albedo = color

    def scatter(self, ray_in, rec, attenuation, scattered):
        scatter_direction = rec.normal + random_unit_vector()
        ray_out = Ray(rec.p, scatter_direction)
        scattered.set(ray_out)
        attenuation.set(self._albedo)
        return True


class Metal:
    def __init__(self, color, fuzz):
        self._albedo = color
        self._fuzz = fuzz

    def scatter(self, ray_in, rec, attenuation, scattered):
        reflected = reflect(Vec3.unit_vector(ray_in.direction()), rec.normal)
        ray_out = Ray(rec.p, reflected + self._fuzz * random_in_unit_sphere())
        scattered.set(ray_out)
        attenuation.set(self._albedo)
        return Vec3.dot(scattered.direction(), rec.normal) > 0


class Dielectric:
    def __init__(self, ref_idx):
        self._ref_idx = ref_idx

    def scatter(self, ray_in, rec, attenuation, scattered):
        attenuation.set(Color(1.0, 1.0, 1.0))
        etai_over_etat = 1.0 / self._ref_idx if rec.front_face else self._ref_idx

        unit_direction = Vec3.unit_vector(ray_in.direction())
        cos_theta = min(Vec3.dot(-unit_direction, rec.normal), 1.0)
        sin_theta = math.sqrt(1.0 - cos_theta**2)
        if etai_over_etat * sin_theta > 1.0:
            reflected = reflect(unit_direction, rec.normal)
            ray_out = Ray(rec.p, reflected)
            scattered.set(ray_out)
            return True

        reflect_prob = schlick(cos_theta, etai_over_etat)
        if random_double() < reflect_prob:
            reflected = reflect(unit_direction, rec.normal)
            ray_out = Ray(rec.p, reflected)
            scattered.set(ray_out)
            return True

        refracted = refract(unit_direction, rec.normal, etai_over_etat)
        ray_out = Ray(rec.p, refracted)
        scattered.set(ray_out)
        return True


class HittableList:
    def __init__(self):
        self._objects = []

    def clear(self):
        self._objects.clear()

    def add(self, obj):
        self._objects.append(obj)

    def hit(self, ray, t_min, t_max, rec):
        temp_rec = HitRecord()
        hit_anything = False

        closest_so_far = t_max
        for obj in self._objects:
            if obj.hit(ray, t_min, closest_so_far, temp_rec):
                hit_anything = True
                closest_so_far = temp_rec.t
                rec.p = temp_rec.p
                rec.t = temp_rec.t
                rec.normal = temp_rec.normal
                rec.front_face = temp_rec.front_face
                rec.material = temp_rec.material

        return hit_anything


class Camera:
    def __init__(self, lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist):
        theta = degrees_to_radians(vfov)
        h = math.tan(theta/2)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        self._w = Vec3.unit_vector(lookfrom - lookat)
        self._u = Vec3.unit_vector(Vec3.cross(vup, self._w))
        self._v = Vec3.cross(self._w, self._u)

        self._origin = lookfrom
        self._horizontal = focus_dist * viewport_width * self._u
        self._vertical = focus_dist * viewport_height * self._v
        self._lower_left_corner = self._origin - self._horizontal / 2 - self._vertical / 2 - focus_dist * self._w

        self._lens_radius = aperture / 2

    def get_ray(self, s, t):
        rd = self._lens_radius * random_in_unit_disk()
        offset = self._u * rd.x() + self._v * rd.y()
        return Ray(self._origin + offset, self._lower_left_corner + s * self._horizontal + t * self._vertical - self._origin - offset)


# constants
infinity = float('inf')
pi = 3.1415926535897932385


# methods
def degrees_to_radians(degrees):
    return degrees * pi / 180


def random_double(min=0.0, max=1.0):
    return min + random.random() * (max - min)


def clamp(value, min, max):
    if value < min:
        return min
    if value > max:
        return max
    return value


def random_in_unit_sphere():
    while True:
        p = Vec3.random(-1.0, 1.0)
        if p.length_squared() >= 1:
            continue
        return p


def random_unit_vector():
    a = random_double(0.0, 2*pi)
    z = random_double(-1.0, 1.0)
    r = math.sqrt(1 - z**2)
    return Vec3(r*math.cos(a), r*math.sin(a), z)


def random_in_hemisphere(normal):
    in_unit_sphere = random_in_unit_sphere()
    if Vec3.dot(in_unit_sphere, normal) > 0.0:
        return in_unit_sphere
    else:
        return -in_unit_sphere


def random_in_unit_disk():
    while True:
        p = Vec3(random_double(-1, 1), random_double(-1, 1), 0)
        if p.length_squared() >= 1:
            continue
        return p


def reflect(v, n):
    return v - 2 * Vec3.dot(v, n) * n


def refract(uv, n, etai_over_etat):
    cos_theta = Vec3.dot(-uv, n)
    r_out_parallel = etai_over_etat * (uv + cos_theta * n)
    r_out_perp = -math.sqrt(1.0 - r_out_parallel.length_squared()) * n
    return r_out_parallel + r_out_perp


def schlick(cosine, ref_idx):
    r0 = (1-ref_idx) / (1+ref_idx)
    r0 = r0**2
    return r0 + (1-r0)*math.pow((1 - cosine), 5)


def color_to_RGB(color):
    def float_to_int(value):
        return int(value * 256)

    return (float_to_int(clamp(math.sqrt(color.x()), 0.0, 0.99)), float_to_int(clamp(math.sqrt(color.y()), 0.0, 0.99)), float_to_int(clamp(math.sqrt(color.z()), 0.0, 0.99)))


def generate_gradient(width, height):
    pixels = []
    for j in range(height - 1, -1, -1):
        for i in range(width):
            pixels.append(Color(i / (width - 1), j / (height - 1), 0.25))
    return pixels


def ray_color(ray, world, depth):
    if depth <= 0:
        return Color(0.0, 0.0, 0.0)

    rec = HitRecord()
    if world.hit(ray, 0.001, infinity, rec):
        scattered = Ray()
        attenuation = Color()
        if rec.material.scatter(ray, rec, attenuation, scattered):
            return attenuation * ray_color(scattered, world, depth - 1)
        return Color(0.0, 0.0, 0.0)

    unit_direction = Vec3.unit_vector(ray.direction())
    t = 0.5*(unit_direction.y() + 1.0)
    return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0)


def render_pixel(data):
    pixel = Color(0, 0, 0)
    for s in range(data["samples"]):
        u = (data["coord"][0] + random_double()) / (data["image_width"] - 1)
        v = (data["coord"][1] + random_double()) / (data["image_height"] - 1)
        r = data["camera"].get_ray(u, v)
        pixel += ray_color(r, data["world"], data["max_depth"])
    return pixel / data["samples"]


if __name__ == "__main__":
    start_time = time.time()

    use_pool = True
    aspect_ratio = 16 / 9
    image_width = 340
    image_height = int(image_width / aspect_ratio)

    # set render quality
    samples = 10
    max_depth = 50

    # setup camera
    look_from = Point3(3, 3, 2)
    look_at = Point3(0, 0, -1)
    vup = Vec3(0, 1, 0)
    dist_to_focus = (look_from - look_at).length()
    aperture = 2.0
    camera = Camera(look_from, look_at, vup, 20, aspect_ratio, aperture, dist_to_focus)

    # create the scene
    world = HittableList()
    world.add(Sphere(Point3(0, 0, -1), 0.5, Lambertian(Color(0.1, 0.2, 0.5))))
    world.add(Sphere(Point3(0, -100.5, -1), 100, Lambertian(Color(0.8, 0.8, 0.0))))

    world.add(Sphere(Point3(1, 0, -1), 0.5, Dielectric(1.5)))
    world.add(Sphere(Point3(1, 0, -1), -0.45, Dielectric(1.5)))
    world.add(Sphere(Point3(-1, 0, -1), 0.5, Metal(Color(0.8, 0.8, 0.8), 1.0)))

    if use_pool:
        # prepare tasks for multiprocess rendering
        tasks = [{"coord": (i, j),
                  "samples": samples,
                  "image_width": image_width,
                  "image_height": image_height,
                  "world": world,
                  "camera": camera,
                  "max_depth": max_depth} for j in range(image_height - 1, -1, -1) for i in range(image_width)]

        # run multiplrocess
        with Pool(8) as render_pool:
            pixels = render_pool.map(render_pixel, tasks)
    else:
        pixels = [render_pixel(
                 {"coord": (i, j),
                  "samples": samples,
                  "image_width": image_width,
                  "image_height": image_height,
                  "world": world,
                  "camera": camera,
                  "max_depth": max_depth}) for j in range(image_height - 1, -1, -1) for i in range(image_width)]

    end_time = time.time()

    # write output file
    with open("image.ppm", "w") as file:
        file.write("P3\n")
        file.write(str(image_width) + " " + str(image_height) + "\n")
        file.write("255\n")
        for pixel in pixels:
            file.write(" ".join([str(v) for v in color_to_RGB(pixel)]) + "\n")

    print("Render time: " + str(end_time - start_time) + " seconds")
