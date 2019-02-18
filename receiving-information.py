# Подключение ZED SDK для работы с камерой
import pyzed.sl as sl

# Печать данных о камере


def print_camera_information(camera):
    print("Resolution: {0}, {1}.".format(round(camera.get_resolution().width, 2), camera.get_resolution().height))
    print("Camera FPS: {0}.".format(camera.get_camera_fps()))
    print("Firmware: {0}.".format(camera.get_camera_information().firmware_version))
    print("Serial number: {0}.\n".format(camera.get_camera_information().serial_number))

# Печать калибровочных данных


def print_calibration_parameters(camera):
    parameters = camera.get_camera_information().calibration_parameters
    print("Left camera:")
    print("Focal lenght: ({0}, {1}),".format(parameters.left_cam.fx, parameters.left_cam.fy))  # Фокусное расстояние
    print("Field of view: ({0}, {1}),".format(parameters.left_cam.h_fov, parameters.left_cam.v_fov))  # Угол обзора
    print("Principal point: ({0}, {1}),".format(parameters.left_cam.cx, parameters.left_cam.cy))  # Координаты принципиальной точки
    # Радиальная дисторсия
    print("Radial distortion: ({0}, {1}, {2}),".format(parameters.left_cam.disto[0], parameters.left_cam.disto[1], parameters.left_cam.disto[4]))
    # Танценциальная дисторсия
    print("Radial distortion: ({0}, {1}),".format(parameters.left_cam.disto[2], parameters.left_cam.disto[3]))

    print("Right camera:")
    print("Focal lenght: ({0}, {1}),".format(parameters.right_cam.fx, parameters.right_cam.fy))  # Фокусное расстояние
    print("Field of view: ({0}, {1}),".format(parameters.right_cam.h_fov, parameters.right_cam.v_fov))  # Угол обзора
    print("Principal point: ({0}, {1}),".format(parameters.right_cam.cx, parameters.right_cam.cy))  # Координаты принципиальной точки
    # Радиальная дисторсия
    print("Radial distortion: ({0}, {1}, {2}),".format(
        parameters.right_cam.disto[0], parameters.right_cam.disto[1], parameters.right_cam.disto[4]))
    # Танценциальная дисторсия
    print("Radial distortion: ({0}, {1}),".format(parameters.right_cam.disto[2], parameters.right_cam.disto[3]))


if __name__ == "__main__":
    init = sl.InitParameters()  # Настроечные параметры камеры
    cam = sl.Camera()  # Экземпляр камеры

    print("Trying to open the camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
    print("The camera opened successfully")

    print_camera_information(cam)
    print_calibration_parameters(cam)

    cam.close()
    print("Finish")
