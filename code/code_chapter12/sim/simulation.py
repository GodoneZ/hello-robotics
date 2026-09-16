"""单个 Isaac World：第十章房间 + 第四章舵轮 + 第九章物理桌面/RGB-D。"""

from pathlib import Path
import math
import numpy as np
from g2_mobile.common import ARM, GRIPPER, load_config, rotation
from g2_mobile.mapping import boxes
from sim.base.base_controller import G2BaseController
from sim.base.config import ControlLimits, RobotGeometry
from sim.base.kinematics import SwerveKinematics
from sim.arm.kinematics import matrix_to_quaternion

ROOT = Path(__file__).resolve().parents[1]


class Simulation:
    def __init__(self, headless=False):
        from isaacsim import SimulationApp

        self.app = SimulationApp(
            {
                "headless": headless,
                "renderer": "RaytracedLighting",
                "multi_gpu": False,
                "active_gpu": 0,
                "physics_gpu": 0,
                "limit_cpu_threads": 12,
                "disable_viewport_updates": False,
                "extra_args": ["--/app/extensions/fsWatcherEnabled=false"],
            }
        )
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.ros2.bridge")
        self.app.update()
        self.cfg = load_config()
        self.dt = 1 / 120
        self._build()

    def _build(self):
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.sensors.camera import Camera
        from pxr import UsdLux
        import omni.usd

        self.world = World(stage_units_in_meters=1.0, physics_dt=self.dt, rendering_dt=1 / 30)
        add_reference_to_stage(
            str(ROOT / "assets/background/room/room_1/background.usda"), "/World"
        )
        self.world.scene.add_default_ground_plane()
        stage = omni.usd.get_context().get_stage()
        light = UsdLux.DomeLight.Define(stage, "/World/TaskLight")
        light.CreateIntensityAttr(1200.0)
        mat = PhysicsMaterial(
            "/World/TaskMaterial", static_friction=1.3, dynamic_friction=1.1, restitution=0.0
        )
        for i, b in enumerate(boxes(self.cfg) + [self.cfg["tray"]]):
            self.world.scene.add(
                FixedCuboid(
                    f"/World/Task/box{i}",
                    name=f"box{i}",
                    position=np.array(b[:3]),
                    scale=np.array(b[3:]),
                    size=1.0,
                    color=np.array([0.43, 0.45, 0.48]) if i == 3 else np.array([0.18, 0.30, 0.34]),
                    physics_material=mat,
                )
            )
        # 完全本地的第九章物体几何：每桌一个苹果及两个干扰物。
        from sim.objects import OBJECTS, mesh_for
        from pxr import UsdGeom, UsdPhysics, UsdShade, Vt, Gf
        from isaacsim.core.prims import SingleRigidPrim

        self.objects = {}
        for station, table in [("source", self.cfg["source_table"]), ("destination", self.cfg["table"])]:
            for item in self.cfg["items"]:
                spec = next(s for s in OBJECTS if s.name == item["name"])
                name = station + "_" + spec.name
                path = "/World/Task/" + name
                root = UsdGeom.Xform.Define(stage, path)
                mesh = UsdGeom.Mesh.Define(stage, path + "/mesh")
                points, faces = mesh_for(spec)
                mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
                mesh.CreateFaceVertexCountsAttr([3] * len(faces))
                mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
                mesh.CreateSubdivisionSchemeAttr("none")
                mesh.CreateDisplayColorAttr([Gf.Vec3f(*spec.color)])
                mesh.CreateDoubleSidedAttr(True)
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                    mat.material, UsdShade.Tokens.weakerThanDescendants, "physics")
                UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
                UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(spec.mass)
                position = [table[0] + item["offset"][0], table[1] + item["offset"][1],
                            table[2] + table[5] / 2 + spec.size[2] / 2 + .008]
                self.objects[name] = self.world.scene.add(SingleRigidPrim(
                    path, name=name, position=np.array(position), mass=spec.mass))
        add_reference_to_stage(str(ROOT / "assets/robot/G2_omnipicker/robot.usda"), "/genie")
        x, y, yaw = self.cfg["start"]
        SingleXFormPrim(
            "/genie",
            position=np.array([x, y, -0.01]),
            orientation=np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]),
        )
        self.world.play()
        for _ in range(120):
            self.world.step(render=True)
        self.robot = self.world.scene.add(SingleArticulation("/genie", name="g2_mobile"))
        self.robot.initialize()
        for body in self.objects.values():
            body.initialize()
        self.robot.set_solver_position_iteration_count(32)
        self.robot.set_solver_velocity_iteration_count(4)
        self.names = list(self.robot.dof_names)
        self.arm_ids = np.array([self.names.index(n) for n in ARM])
        self.grip_id = self.names.index(GRIPPER)
        self.left_ids = np.array([self.names.index(f"idx2{i}_arm_l_joint{i}") for i in range(1, 8)])
        self.robot.set_joint_positions(np.array(self.cfg["home"]), joint_indices=self.arm_ids)
        self.robot.set_joint_positions(np.array(self.cfg["left_home"]), joint_indices=self.left_ids)
        self.arm_target = np.array(self.cfg["home"])
        self.grip_target = 0.785
        self.base = G2BaseController(
            self.robot,
            SwerveKinematics(RobotGeometry().wheel_positions, RobotGeometry().wheel_radius),
            ControlLimits(),
        )
        self.base_prim = SingleXFormPrim("/genie/base_link")
        self.arm_prim = SingleXFormPrim("/genie/arm_base_link")
        self.tcp_prim = SingleXFormPrim("/genie/gripper_r_center_link")
        self.camera = Camera(
            "/World/TaskCamera", name="task_rgbd", frequency=30, resolution=(320, 240)
        )
        self.move_camera()
        self.camera.set_focal_length(24.0)
        self.camera.set_horizontal_aperture(26.0)
        self.camera.set_vertical_aperture(19.5)
        self.camera.set_clipping_range(0.05, 5.0)
        self.camera.initialize()
        self.camera.add_rgb_to_frame()
        self.camera.add_distance_to_image_plane_to_frame()
        for _ in range(120):
            self.step((0, 0, 0))
        print("[chapter12] base", self.pose(), "arm", self.arm_prim.get_world_pose(), flush=True)

    @property
    def time(self):
        return float(self.world.current_time)

    def pose(self):
        p, q = self.base_prim.get_world_pose()
        w, x, y, z = q
        return np.array([p[0], p[1], math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])

    def move_camera(self):
        # 第九章俯视 RGB-D 的安装偏置随底盘移动，不读取物体真值。
        p, q = self.arm_prim.get_world_pose()
        R = rotation([q[1], q[2], q[3], q[0]])
        target = p + R @ np.array([0.57, 0.45, -0.20])
        # 稍偏侧面的俯视，避免手掌在抬升验证时完全挡住苹果。
        eye = target + R @ np.array([0.70, -0.30, 0.15])
        z = (target - eye) / np.linalg.norm(target - eye)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        Rc = np.column_stack([x, np.cross(z, x), z])
        self.camera.set_world_pose(eye, matrix_to_quaternion(Rc), camera_axes="ros")

    def hold(self):
        self.arm_target = np.asarray(self.robot.get_joint_positions())[self.arm_ids].copy()
        # 闭合驱动是持物力的来源，IDLE/心跳超时只停止臂和底盘，不能
        # 把闭合目标 0 改成被物体挡住的实测开度（约 .25），否则失去夹紧力。
        # 未完成的开合仍冻结；显式 GripperCommand 才能松开已夹持的物体。
        if self.grip_target > .1:
            self.grip_target = float(self.robot.get_joint_positions()[self.grip_id])

    def step(self, velocity):
        from isaacsim.core.utils.types import ArticulationAction

        self.base.set_velocity(*velocity)
        self.base.update(float(self.world.get_rendering_dt()))
        self.robot.apply_action(
            ArticulationAction(joint_positions=self.arm_target, joint_indices=self.arm_ids)
        )
        self.robot.apply_action(
            ArticulationAction(
                joint_positions=np.array(self.cfg["left_home"]), joint_indices=self.left_ids
            )
        )
        self.robot.apply_action(
            ArticulationAction(
                joint_positions=np.array([self.grip_target]), joint_indices=np.array([self.grip_id])
            )
        )
        self.move_camera()
        self.world.step(render=True)

    def close(self):
        if hasattr(self, "base"):
            self.base.stop()
        self.app.close()
