#!/usr/bin/env python3
"""Chapter 12 object perception.

The detector is YOLO (Chapter 9 weights copied into this package).  The RGB-D
point cloud is only used to refine the detected apple centre and generate the
geometric grasp poses; it is never used as a simulator ground-truth source.
Missing YOLO is a hard perception fault; no colour or ground-truth fallback.
"""
import os
import time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseArray, Pose, PointStamped
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from tf2_ros import Buffer, TransformListener, TransformException
from g2_mobile.common import load_config, transform, grasp_candidates, rotation

class Perception(Node):
    def __init__(self):
        super().__init__('yolo_rgbd_perception')
        self.cfg = load_config()
        self.station = self.declare_parameter('station', 'auto').value
        self.tf = Buffer(); self.listener = TransformListener(self.tf, self)
        self.candidates = self.create_publisher(PoseArray, '/grasp_candidates', 10)
        self.center = self.create_publisher(PointStamped, '/target_center', 10)
        self.create_subscription(PointCloud2, '/camera/points', self.on_cloud, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/color/image_raw', self.on_rgb, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/depth/image_raw', self.on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_info, qos_profile_sensor_data)
        self.rgb = None; self.depth = None; self.info = None
        self.frames = {key: {} for key in ('cloud', 'rgb', 'depth', 'info')}
        self.phase = 'START'
        self.create_subscription(String, '/task_status', self.on_status,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.last_diagnostic = -float("inf")
        self.last_result = "启动"
        self.model = None; self.model_error = None
        self._load_yolo()
        self.create_timer(0.08, self.process)

    def _load_yolo(self):
        # ROS 节点使用系统 Python 中通过 pip 安装的固定版本。
        root = Path(__file__).resolve().parents[1]
        os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/g2_chapter12_ultralytics')
        path = root / 'models' / 'apple_yolo.pt'
        if not path.is_file():
            try:
                from ament_index_python.packages import get_package_share_directory
                path = Path(get_package_share_directory('g2_chapter12')) / 'models' / 'apple_yolo.pt'
            except Exception:
                pass
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(path))
            self.get_logger().info(f'YOLO apple detector loaded: {path}')
        except Exception as exc:
            self.model_error = str(exc)
            self.get_logger().warning('YOLO unavailable; grasp disabled (no fallback): ' + repr(exc))

    def diagnostic(self, message):
        # 墙钟节流：即使 /clock 未启动，也能报告传感器缺失。
        now = time.monotonic()
        if now - self.last_diagnostic >= 3.0:
            self.last_diagnostic = now
            self.get_logger().info('[perception] ' + message)

    def on_status(self, msg):
        self.phase = msg.data.split(':')[0]

    def cache(self, kind, msg):
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.frames[kind][stamp] = msg
        while len(self.frames[kind]) > 8:
            del self.frames[kind][min(self.frames[kind])]

    def on_cloud(self, msg): self.cache('cloud', msg)
    def on_rgb(self, msg): self.cache('rgb', msg)
    def on_depth(self, msg): self.cache('depth', msg)
    def on_info(self, msg): self.cache('info', msg)

    def _image(self, msg):
        if msg.encoding in ('rgb8', 'bgr8'):
            a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width*3]
            a = a.reshape(msg.height, msg.width, 3)
            return a[..., ::-1] if msg.encoding == 'rgb8' else a
        return None

    def detect_apples(self, image, rotated=False):
        """夹持后物体方向会改变：只在 LIFT 用四个直角视图做 YOLO 检测。

        检测框严格还原至原图像素，仍需真实深度与 TCP 邻域校验。
        不降低置信度，不用颜色或夹爪自身的点云冒充苹果。
        """
        h, w = image.shape[:2]
        apples = []
        for k in (range(4) if rotated else (0,)):
            view = np.ascontiguousarray(np.rot90(image, k))
            result = self.model.predict(
                view, conf=float(self.cfg.get('vision', {}).get('confidence', .4)),
                verbose=False, device='cpu')[0]
            for box in result.boxes:
                if str(result.names[int(box.cls[0])]).lower() != 'apple':
                    continue
                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
                if k == 1:
                    coords = [w-y2, x1, w-y1, x2]
                elif k == 2:
                    coords = [w-x2, h-y2, w-x1, h-y1]
                elif k == 3:
                    coords = [y1, h-x2, y2, h-x1]
                else:
                    coords = [x1, y1, x2, y2]
                apples.append((np.asarray(coords), float(box.conf[0])))
        return apples

    def detect_held_apples(self, image, goal, camera_tf):
        """以同帧 TCP 投影选取局部视图，仍须 YOLO 识别及原图真实深度验证。

        全图中夹持苹果过小或方向改变时，局部视图保留更多目标细节。
        TCP 只定位搜索窗口，不提供苹果坐标，也不作为检测成功的依据。
        """
        tr, q = camera_tf.translation, camera_tf.rotation
        camera = rotation([q.x, q.y, q.z, q.w]).T @ (np.asarray(goal) - [tr.x, tr.y, tr.z])
        if not np.isfinite(camera).all() or camera[2] <= .12:
            return []
        fx, fy, cx, cy = self.info.k[0], self.info.k[4], self.info.k[2], self.info.k[5]
        u, v = fx * camera[0] / camera[2] + cx, fy * camera[1] / camera[2] + cy
        # 12 cm 搜索邻域，同时保留至少 60% 图宽的上下文；过紧裁剪会
        # 让模型只见夹爪碎片而漏检。裁剪不改变后续 8 cm 真实深度筛选。
        h, w = image.shape[:2]
        rx = max(w * .30, fx * .12 / (camera[2] - .12))
        ry = max(w * .30, fy * .12 / (camera[2] - .12))
        x1, x2 = np.clip([np.floor(u-rx), np.ceil(u+rx)], 0, w).astype(int)
        y1, y2 = np.clip([np.floor(v-ry), np.ceil(v+ry)], 0, h).astype(int)
        if x2-x1 < 16 or y2-y1 < 16:
            return []
        offset = np.array([x1, y1, x1, y1])
        return [(coords + offset, conf) for coords, conf in
                self.detect_apples(image[y1:y2, x1:x2], rotated=True)]

    def _yolo_points(self, table, top):
        """用 YOLO 框内真实深度反投影苹果点云；不以重复中心点代替几何。"""
        if self.model is None or self.rgb is None or self.depth is None or self.info is None:
            return None, 'RGB-D/YOLO 输入未就绪'
        image = self._image(self.rgb)
        if image is None:
            return None, f'不支持 RGB 编码: {self.rgb.encoding}'
        try:
            if self.depth.encoding in ('32FC1', '32FC'):
                dtype = np.float32
            elif self.depth.encoding in ('16UC1', '16UC'):
                dtype = np.uint16
            else:
                return None, f'不支持深度编码: {self.depth.encoding}'
            depth = np.frombuffer(self.depth.data, dtype=dtype).reshape(self.depth.height, self.depth.step // np.dtype(dtype).itemsize)[:, :self.depth.width]
            if dtype == np.uint16:
                depth = depth.astype(np.float32) * 0.001
            stamp = rclpy.time.Time.from_msg(self.depth.header.stamp)
            tf = self.tf.lookup_transform('odom', self.depth.header.frame_id, stamp)
            q, tr = tf.transform.rotation, tf.transform.translation
            tracking_lift = self.phase.endswith('_LIFT')
            verify_place = self.phase.endswith(('_RETREAT', '_VERIFY_PLACE'))
            goal = None
            if tracking_lift:
                tcp = self.tf.lookup_transform('odom', 'gripper_r_center_link', stamp).transform.translation
                goal = np.array([tcp.x, tcp.y, tcp.z])
            elif verify_place:
                index = 0 if self.phase.startswith('SOURCE') else 1
                goal = np.r_[self.cfg['place_slots'][index], top+.03]
            boxes = self.detect_apples(image, rotated=tracking_lift)
            if tracking_lift:
                boxes += self.detect_held_apples(image, goal, tf.transform)
            choices = []
            details = []
            for coords, conf in boxes:
                x1,y1,x2,y2 = coords.astype(int)
                x1,x2 = np.clip([x1,x2],0,depth.shape[1]-1)
                y1,y2 = np.clip([y1,y2],0,depth.shape[0]-1)
                yy,xx = np.mgrid[y1:y2+1,x1:x2+1]
                z = depth[y1:y2+1,x1:x2+1]
                valid = np.isfinite(z) & (z>.05) & (z<5.)
                fx,fy,cx,cy = self.info.k[0],self.info.k[4],self.info.k[2],self.info.k[5]
                cam = np.column_stack(((xx[valid]-cx)*z[valid]/fx,
                                       (yy[valid]-cy)*z[valid]/fy,z[valid]))
                pts = transform(cam,[tr.x,tr.y,tr.z],[q.x,q.y,q.z,q.w])
                details.append(f'conf={conf:.2f} raw={len(pts)} '
                               f'median={np.round(np.median(pts,axis=0),3).tolist()} goal={goal}')
                if tracking_lift:
                    pts = pts[np.linalg.norm(pts-goal,axis=1)<.08]
                else:
                    pts = pts[(abs(pts[:,0]-table[0])<table[3]/2) &
                              (abs(pts[:,1]-table[1])<table[4]/2) &
                              (pts[:,2]>top+.006) & (pts[:,2]<top+.14)]
                if len(pts)<30: continue
                med=np.median(pts,axis=0)
                pts=pts[np.linalg.norm(pts[:,:2]-med[:2],axis=1)<.06]
                if len(pts)<30: continue
                med=np.median(pts,axis=0)
                tray=self.cfg['tray']
                # 第二次抓取只选板外苹果，不能把刚放下的第一颗又抓走。
                if self.phase == 'DESTINATION_PERCEIVE' and (
                        abs(med[0]-tray[0])<tray[3]/2+.015 and
                        abs(med[1]-tray[1])<tray[4]/2+.015):
                    continue
                score = -np.linalg.norm(med-goal) if goal is not None else conf
                if goal is not None and np.linalg.norm(med-goal)>.10: continue
                choices.append((score,pts,conf))
            if not choices and verify_place:
                # 放置复核不强制 YOLO 再次识别：板上苹果可能被夹爪/板边遮挡。
                # 仍只使用当前同步 RGB-D 深度，在指定槽位的真实几何 ROI 内复核。
                yy, xx = np.mgrid[:depth.shape[0], :depth.shape[1]]
                valid = np.isfinite(depth) & (depth > .05) & (depth < 5.)
                fx,fy,cx,cy = self.info.k[0],self.info.k[4],self.info.k[2],self.info.k[5]
                z = depth[valid]
                cam = np.column_stack(((xx[valid]-cx)*z/fx, (yy[valid]-cy)*z/fy, z))
                cloud = transform(cam,[tr.x,tr.y,tr.z],[q.x,q.y,q.z,q.w])
                board_z = tray[2] + tray[5] / 2.0
                near = cloud[(np.linalg.norm(cloud[:,:2]-goal[:2],axis=1)<.075) &
                             (cloud[:,2]>board_z+.010) & (cloud[:,2]<board_z+.14)]
                if len(near) >= 20:
                    med = np.median(near,axis=0)
                    near = near[np.linalg.norm(near-med,axis=1)<.055]
                    if len(near) >= 20:
                        return (near, 1.0), 'RGB-D 槽位几何复核（YOLO 遮挡）'
            if not choices:
                return None, self.phase + ': 无有效苹果 ' + '; '.join(details)
            _,pts,conf=max(choices,key=lambda item:item[0])
            return (pts,conf), f'YOLO apple conf={conf:.2f}'
        except Exception as exc:
            self.get_logger().debug('YOLO RGB-D frame skipped: ' + repr(exc))
            return None, 'YOLO/RGB-D 处理异常: ' + str(exc)

    def ready_stamp(self):
        """Retain images until all same-stamp TFs arrive (including TCP).

        robot_state_publisher may lag the camera by a physics tick. Do not use
        latest TF or repeatedly discard each new image before TCP TF arrives.
        """
        stamps = set.intersection(*(set(frames) for frames in self.frames.values()))
        for stamp in sorted(stamps, reverse=True):
            depth = self.frames['depth'][stamp]
            cloud = self.frames['cloud'][stamp]
            at = rclpy.time.Time.from_msg(depth.header.stamp)
            frames = {depth.header.frame_id, cloud.header.frame_id}
            if self.phase.endswith('_LIFT'):
                frames.add('gripper_r_center_link')
            if all(self.tf.can_transform('odom', frame, at) for frame in frames):
                return stamp
        return None

    def process(self):
        stamp = self.ready_stamp()
        if stamp is None:
            self.diagnostic('等待同步帧及同时间戳 TF；上次结果: ' + self.last_result)
            return
        self.rgb,self.depth,self.info,msg=(self.frames[k][stamp] for k in ('rgb','depth','info','cloud'))
        for frames in self.frames.values():
            for old in list(frames):
                if old<=stamp: del frames[old]
        if (msg.point_step != 16 or msg.is_bigendian or msg.height != 1
                or len(msg.data) != msg.width * 16):
            self.diagnostic('点云格式不匹配：需要 little-endian XYZ/RGB，point_step=16')
            return
        age = (self.get_clock().now() - rclpy.time.Time.from_msg(msg.header.stamp)).nanoseconds / 1e9
        if age < -0.1 or age > 0.5:
            self.diagnostic(f'点云时间不新鲜 age={age:.3f}s，检查 /clock 与传感器频率')
            return
        try:
            self.tf.lookup_transform('odom', msg.header.frame_id, rclpy.time.Time.from_msg(msg.header.stamp))
        except TransformException as exc:
            self.diagnostic(f'等待点云时间戳对应的 odom TF: {exc}')
            return
        # Validate the cloud/TF stream, then use dense RGB-D pixels selected by YOLO.
        if self.station == 'source':
            table=np.asarray(self.cfg['source_table'],float)
        elif self.station == 'destination':
            table=np.asarray(self.cfg['table'],float)
        else:
            # 自动选择离机器人底盘最近的桌子，支持同一节点完成两站任务。
            try:
                bt=self.tf.lookup_transform('odom','base_link',rclpy.time.Time())
                bx,by=bt.transform.translation.x,bt.transform.translation.y
                candidates=(np.asarray(self.cfg['source_table'],float),np.asarray(self.cfg['table'],float))
                table=min(candidates,key=lambda x:(x[0]-bx)**2+(x[1]-by)**2)
            except TransformException:
                table=np.asarray(self.cfg['source_table'],float)
        top=table[2]+table[5]/2
        detected, reason = self._yolo_points(table, top)
        self.last_result = reason
        if detected is None:
            self.diagnostic(reason)
            return
        points, confidence = detected
        if len(points) < 30:
            self.diagnostic(f'真实 RGB-D 苹果点不足：{len(points)} / 30')
            return
        center=np.median(points,axis=0)

        if max(np.ptp(points[:,:2],axis=0))>.14:
            self.diagnostic('目标点云跨度过大，可能混入多个物体，拒绝合并抓取')
            return
        obs=PointStamped(); obs.header=msg.header; obs.header.frame_id='odom'; obs.point.x,obs.point.y,obs.point.z=map(float,center); self.center.publish(obs)
        poses=PoseArray(); poses.header=obs.header
        for xyzq in grasp_candidates(points,top):
            p=Pose(); p.position.x,p.position.y,p.position.z=map(float,xyzq[:3]); p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w=map(float,xyzq[3:]); poses.poses.append(p)
        self.candidates.publish(poses)
        self.diagnostic(f'phase={self.phase} station={self.station} points={len(points)} '
                        f'center={np.round(center, 3).tolist()} grasps={len(poses.poses)} '
                        f'backend=YOLO')

def main():
    rclpy.init(); n=Perception()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally: n.destroy_node(); rclpy.shutdown() if rclpy.ok() else None
if __name__=='__main__': main()
