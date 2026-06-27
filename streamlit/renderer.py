import cv2
import math


class OverlayRenderer:

    def draw_detection_boxes(
            self,
            image,
            filtered_boxes,
            xyxy,
            confs,
            show_debug=True
    ):

        for idx in filtered_boxes:

            x1, y1, x2, y2 = map(
                int,
                xyxy[idx]
            )

            conf = float(confs[idx])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                f"Seal {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            if show_debug:
                cv2.circle(
                    image,
                    (cx, cy),
                    8,
                    (255, 255, 0),
                    -1
                )

        return image

    def draw_anchor(
            self,
            image,
            anchor
    ):

        if anchor is None:
            return image

        ax, ay = anchor["center"]

        cv2.circle(
            image,
            (ax, ay),
            14,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            image,
            "ANCHOR",
            (ax + 15, ay),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        return image

    def draw_match(
            self,
            image,
            anchor,
            candidate
    ):

        if (
                anchor is None
                or
                candidate is None
        ):
            return image

        ax, ay = anchor["center"]

        bx, by = candidate["center"]

        cv2.circle(
            image,
            (bx, by),
            14,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            image,
            "MATCH",
            (bx + 15, by),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.line(
            image,
            (ax, ay),
            (bx, by),
            (0, 255, 0),
            4
        )

        return image

    def draw_cone(
            self,
            image,
            anchor,
            direction,
            max_angle
    ):

        if anchor is None:
            return image

        ax, ay = anchor["center"]

        cone_length = 450

        center_angle = (
            0
            if direction == "RIGHT"
            else 180
        )

        upper_angle = math.radians(
            center_angle + max_angle
        )

        lower_angle = math.radians(
            center_angle - max_angle
        )

        upper_x = int(
            ax +
            cone_length *
            math.cos(upper_angle)
        )

        upper_y = int(
            ay -
            cone_length *
            math.sin(upper_angle)
        )

        lower_x = int(
            ax +
            cone_length *
            math.cos(lower_angle)
        )

        lower_y = int(
            ay -
            cone_length *
            math.sin(lower_angle)
        )

        cv2.line(
            image,
            (ax, ay),
            (upper_x, upper_y),
            (0, 255, 255),
            2
        )

        cv2.line(
            image,
            (ax, ay),
            (lower_x, lower_y),
            (0, 255, 255),
            2
        )

        return image

    def draw_angle_distance(
            self,
            image,
            anchor,
            angle,
            distance
    ):

        if anchor is None:
            return image

        ax, ay = anchor["center"]

        cv2.putText(
            image,
            f"Angle: {angle:.2f}",
            (ax + 20, ay - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"Distance: {distance:.2f}",
            (ax + 20, ay),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        return image

    def draw_dashboard(
            self,
            image,
            result
    ):

        overlay = image.copy()

        cv2.rectangle(
            overlay,
            (10, 10),
            (450, 260),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.75,
            image,
            0.25,
            0,
            image
        )

        meter_color = (
            (255, 0, 0)
            if result["suspicious"]
            else
            (0, 255, 0)
        )

        cv2.putText(
            image,
            result["status"],
            (30, 55),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            meter_color,
            3
        )

        rows = [

            (
                "TOP SEAL",
                result["top_present"]
            ),

            (
                "BOTTOM LEFT",
                result["bottom_left_present"]
            ),

            (
                "BOTTOM RIGHT",
                result["bottom_right_present"]
            )
        ]

        start_y = 120

        for i, (label, present) in enumerate(rows):
            y = start_y + (i * 50)

            color = (
                (0, 255, 0)
                if present
                else
                (255, 0, 0)
            )

            state = (
                "PRESENT"
                if present
                else
                "MISSING"
            )

            cv2.putText(
                image,
                f"{label} : {state}",
                (30, y),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                color,
                2
            )

        return image