import cv2
import numpy as np


def create_kalman_constant_velocity(dt: float = 1.0) -> cv2.KalmanFilter:
    """
    Constant Velocity Kalman Filter (2D):
      state:       [x, y, vx, vy]^T
      measurement: [x, y]^T
    """
    kf = cv2.KalmanFilter(4, 2)

    kf.measurementMatrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.float32,
    )

    # Ковариация ошибки состояния.
    # Большие значения -> фильтр больше доверяет измерению.
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 500.0

    # Текущее состояние фильтра после коррекции.
    kf.statePost = np.zeros((4, 1), dtype=np.float32)

    # Шум модели движения.
    # Большое значение:
    #   фильтр быстрее реагирует на изменения
    #   меньше доверяет модели
    #
    # Малое значение:
    #   движение становится очень плавным
    #   но появляется запаздывание
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    # Шум измерения
    # Большое значение:
    #   фильтр меньше доверяет измерению
    #   сильнее сглаживает
    #
    # Малое значение:
    #   фильтр почти повторяет измерение
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0

    update_transition_matrix_cv(kf, dt)
    return kf


def update_transition_matrix_cv(kf: cv2.KalmanFilter, dt: float) -> None:
    """
    transitionMatrix (F)

    Описывает, как состояние меняется со временем:
        x'  = x + vx*dt
        y'  = y + vy*dt
        vx' = vx
        vy' = vy
    """
    dt = float(max(1e-3, dt))
    kf.transitionMatrix = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def pick_largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру (VideoCapture(0)).")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    kf = create_kalman_constant_velocity(dt=1.0)
    kf_initialized = False

    prev_tick = cv2.getTickCount()

    measurement_noise_std = 15.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        tick = cv2.getTickCount()
        dt = (tick - prev_tick) / cv2.getTickFrequency()
        prev_tick = tick
        update_transition_matrix_cv(kf, dt)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        face = pick_largest_face(faces)


        pred = kf.predict()
        pred_x, pred_y = float(pred[0, 0]), float(pred[1, 0])

        noisy_point = None

        if face is not None:
            x, y, w, h = face
            cx = x + w / 2.0
            cy = y + h / 2.0

            # Добавляем искусственный шум в измерение
            # чтобы показать работу фильтра
            noisy_cx = cx + np.random.normal(0, measurement_noise_std)
            noisy_cy = cy + np.random.normal(0, measurement_noise_std)
            noisy_point = (noisy_cx, noisy_cy)

            if not kf_initialized:
                kf.statePost = np.array(
                    [[noisy_cx], [noisy_cy], [0.0], [0.0]], dtype=np.float32
                )
                kf_initialized = True
            else:
                measurement = np.array([[noisy_cx], [noisy_cy]], dtype=np.float32)
                kf.correct(measurement)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.circle(frame, (int(pred_x), int(pred_y)), 6, (255, 0, 0), -1)

        if noisy_point is not None:
            cv2.circle(frame, (int(noisy_point[0]), int(noisy_point[1])), 6, (0, 0, 255), -1)

        if kf_initialized:
            est = kf.statePost
            est_x, est_y = float(est[0, 0]), float(est[1, 0])
            cv2.circle(frame, (int(est_x), int(est_y)), 6, (0, 255, 255), -1)

            vx = float(est[2, 0])
            vy = float(est[3, 0])
            cv2.putText(
                frame,
                f"vx={vx:+.1f} vy={vy:+.1f}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            "Red: measurement | Blue: KF predict | Yellow: KF estimate",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Kalman Filter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
