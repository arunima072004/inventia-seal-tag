import math


class SealClassifier:

    def __init__(
            self,
            max_angle,
            min_bottom_distance
    ):

        self.max_angle = max_angle
        self.min_bottom_distance = min_bottom_distance

    def split_regions(
            self,
            seal_centers,
            image_width,
            image_height
    ):

        top_candidates = []
        bottom_left_candidates = []
        bottom_right_candidates = []

        if len(seal_centers) == 0:
            return (
                top_candidates,
                bottom_left_candidates,
                bottom_right_candidates
            )

        left_boundary = image_width // 2

        lowest_seal_y = max(
            s["center"][1]
            for s in seal_centers
        )

        bottom_band_limit = (
                lowest_seal_y -
                (image_height * 0.15)
        )

        for s in seal_centers:

            sx, sy = s["center"]

            if sy >= bottom_band_limit:

                if sx < left_boundary:
                    bottom_left_candidates.append(s)
                else:
                    bottom_right_candidates.append(s)

            else:

                top_candidates.append(s)

        return (
            top_candidates,
            bottom_left_candidates,
            bottom_right_candidates
        )

    def choose_anchor(
            self,
            bottom_left_candidates,
            bottom_right_candidates
    ):

        if (
                len(bottom_left_candidates) == 0
                or
                len(bottom_right_candidates) == 0
        ):
            return None, None, None

        best_left = max(
            bottom_left_candidates,
            key=lambda x: x["conf"]
        )

        best_right = max(
            bottom_right_candidates,
            key=lambda x: x["conf"]
        )

        if best_left["conf"] >= best_right["conf"]:
            return (
                best_left,
                bottom_right_candidates,
                "RIGHT"
            )

        return (
            best_right,
            bottom_left_candidates,
            "LEFT"
        )

    def validate_alignment(
            self,
            anchor,
            search_candidates,
            direction
    ):

        if anchor is None:
            return {
                "matched": False,
                "angle": 0,
                "distance": 0,
                "candidate": None
            }

        ax, ay = anchor["center"]

        best_candidate = None
        best_conf = -1

        best_angle = 0
        best_distance = 0

        for candidate in search_candidates:

            if candidate == anchor:
                continue

            cx, cy = candidate["center"]

            dx = cx - ax
            dy = ay - cy

            distance = math.sqrt(
                dx ** 2 + dy ** 2
            )

            if distance < self.min_bottom_distance:
                continue

            candidate_angle = abs(
                math.degrees(
                    math.atan2(dy, dx)
                )
            )

            if direction == "LEFT":
                candidate_angle = abs(
                    180 - candidate_angle
                )

            # Must lie inside angular cone
            if candidate_angle > self.max_angle:
                continue

            # Choose highest-confidence candidate
            if candidate["conf"] > best_conf:
                best_conf = candidate["conf"]

                best_candidate = candidate

                best_angle = candidate_angle

                best_distance = distance

        return {
            "matched": best_candidate is not None,
            "angle": best_angle,
            "distance": best_distance,
            "candidate": best_candidate,
            "anchor_x": ax,
            "anchor_y": ay
        }

    def classify_meter(
            self,
            seal_centers,
            image_width,
            image_height
    ):

        result = {
            "status": "NO SEALS DETECTED",
            "reason": "NO_SEALS",
            "suspicious": True,
            "angle": 0,
            "distance": 0
        }

        (
            top_candidates,
            bottom_left_candidates,
            bottom_right_candidates
        ) = self.split_regions(
            seal_centers,
            image_width,
            image_height
        )

        top_present = (
                len(top_candidates) > 0
        )

        bottom_left_present = (
                len(bottom_left_candidates) > 0
        )

        bottom_right_present = (
                len(bottom_right_candidates) > 0
        )

        result["top_present"] = top_present
        result["bottom_left_present"] = bottom_left_present
        result["bottom_right_present"] = bottom_right_present

        result["main_seals_detected"] = (
                int(top_present)
                +
                int(bottom_left_present)
                +
                int(bottom_right_present)
        )

        result["extra_seals"] = max(
            0,
            len(seal_centers)
            -
            result["main_seals_detected"]
        )

        anchor, search_candidates, direction = (
            self.choose_anchor(
                bottom_left_candidates,
                bottom_right_candidates
            )
        )

        result["direction"] = direction

        alignment = self.validate_alignment(
            anchor,
            search_candidates,
            direction
        )

        result["angle"] = alignment["angle"]
        result["distance"] = alignment["distance"]

        suspicious = False
        reason = "VALID"

        if not top_present:

            suspicious = True
            reason = "TOP_SEAL_MISSING"

        elif not bottom_left_present:

            suspicious = True
            reason = "BOTTOM_LEFT_MISSING"

        elif not bottom_right_present:

            suspicious = True
            reason = "BOTTOM_RIGHT_MISSING"

        elif not alignment["matched"]:

            suspicious = True
            reason = "NO_VALID_OPPOSITE_SEAL"

        result["suspicious"] = suspicious

        result["reason"] = reason

        result["status"] = (
            "TAMPER SUSPECTED"
            if suspicious
            else
            "VALID METER"
        )

        result["anchor"] = anchor

        result["matched_candidate"] = (
            alignment["candidate"]
        )

        return result