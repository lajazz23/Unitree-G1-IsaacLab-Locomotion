import omni
from pxr import Gf

def get_vr_controller_pose():
    viewport = omni.kit.viewport.get_viewport_interface()
    if not viewport:
        return None, None

    devices = viewport.get_tracked_devices()
    if not devices:
        return None, None

    controller_id = None
    for device in devices:
        if "controller" in device.lower():
            controller_id = device
            break

    if not controller_id:
        controller_id = devices[0]

    mat = viewport.get_tracked_device_transform(controller_id)

    if mat is None:
        return None, None

    pos = mat.ExtractTranslation() 

    forward = -mat.ExtractRotation().GetInverse().GetAxisAngle()[0]
    forward_vec = Gf.Vec3d(-mat.GetColumn(2)[0], -mat.GetColumn(2)[1], -mat.GetColumn(2)[2])

    return (pos[0], pos[1], pos[2]), (forward_vec[0], forward_vec[1], forward_vec[2])
