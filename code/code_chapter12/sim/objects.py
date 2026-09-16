"""第九章教学物体的本地副本：几何生成仅用于场景，不向感知提供位姿。"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ObjectSpec:
    name: str
    prompt: str
    shape: str
    size: tuple  # X,Y,Z 包围盒（m）
    color: tuple
    mass: float = .035

OBJECTS = (
    ObjectSpec("apple", "a small red apple", "apple", (.050, .048, .050), (.78, .05, .025)),
    ObjectSpec("orange", "an orange fruit", "sphere", (.047, .047, .047), (.95, .32, .015)),
    ObjectSpec("lemon", "a yellow lemon", "ellipsoid", (.066, .039, .039), (.92, .80, .025)),
    ObjectSpec("tomato", "a red tomato", "sphere", (.046, .046, .038), (.90, .07, .035)),
    ObjectSpec("potato", "a small potato", "ellipsoid", (.068, .043, .040), (.49, .31, .12)),
    ObjectSpec("soap", "a bar of soap", "box", (.070, .038, .027), (.82, .58, .71)),
    ObjectSpec("sponge", "a kitchen sponge", "box", (.070, .040, .030), (.93, .78, .09)),
    ObjectSpec("tea_box", "a small cardboard tea box", "box", (.060, .040, .055), (.17, .46, .22)),
    ObjectSpec("can", "a small soda can", "can", (.040, .040, .073), (.70, .10, .04)),
    ObjectSpec("bottle", "a small opaque plastic bottle", "bottle", (.038, .038, .085), (.10, .35, .70)),
    ObjectSpec("cup", "a small cup", "cup", (.050, .050, .060), (.78, .82, .90)),
    ObjectSpec("spool", "a spool of thread", "spool", (.045, .045, .050), (.20, .62, .60)),
    ObjectSpec("wood_block", "a wooden toy block", "box", (.042, .042, .042), (.62, .38, .16)),
    ObjectSpec("eraser", "a rectangular eraser", "box", (.060, .026, .026), (.32, .53, .80)),
    ObjectSpec("battery", "a cylindrical battery", "can", (.024, .024, .050), (.12, .12, .13)),
)

def lathe(profile, sides=40):
    """绕 Z 轴旋转截面，构造三角网格（瓶、杯、线轴）。"""
    a = np.arange(sides)*2*np.pi/sides
    points = np.array([[r*np.cos(t), r*np.sin(t), z] for r,z in profile for t in a])
    faces = []
    for j in range(len(profile)-1):
        for i in range(sides):
            u, v = j*sides+i, j*sides+(i+1)%sides
            faces.extend([(u,v,v+sides), (u,v+sides,u+sides)])
    return points, np.asarray(faces)


def mesh_for(spec):
    sx,sy,sz = spec.size
    if spec.shape in ('sphere','ellipsoid','apple'):
        t = np.linspace(-np.pi/2, np.pi/2, 25)
        profile = [(max(1e-6, np.cos(a)/2), np.sin(a)/2) for a in t]
        p,f = lathe(profile)
        if spec.shape == 'apple':
            p[:,2] *= .92 + .08*np.cos(np.arctan2(p[:,1],p[:,0])*5)
        return p*np.array([sx,sy,sz]), f
    if spec.shape == 'box':
        p = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])*.5
        f = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
        return p*np.array(spec.size), f
    profiles = {
        'can': [(0,-.5),(.48,-.5),(.5,-.47),(.5,.47),(.48,.5),(0,.5)],
        'bottle': [(0,-.5),(.47,-.5),(.5,-.46),(.5,.18),(.26,.32),(.26,.45),(.29,.45),(.29,.5),(0,.5)],
        'spool': [(0,-.5),(.5,-.5),(.5,-.38),(.31,-.38),(.31,.38),(.5,.38),(.5,.5),(0,.5)],
        'cup': [(0,-.5),(.40,-.5),(.5,.5),(.44,.5),(.34,-.40),(0,-.40)],
    }
    p,f = lathe(profiles[spec.shape])
    return p*np.array(spec.size), f


