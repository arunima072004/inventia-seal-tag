import pandas as pd
import zipfile
import os


class ReportGenerator:

    def __init__(self):
        self.rows = []

    def add_record(
        self,
        image_name,
        result,
        detected_seals
    ):

        self.rows.append({

            "image_name": image_name,

            "detected_seals": detected_seals,

            "angle": round(
                result["angle"],
                2
            ),

            "distance": round(
                result["distance"],
                2
            ),

            "top_present":
                result["top_present"],

            "bottom_left_present":
                result["bottom_left_present"],

            "bottom_right_present":
                result["bottom_right_present"],

            "main_seals_detected":
                result["main_seals_detected"],

            "extra_seals":
                result["extra_seals"],

            "classification":
                result["status"],

            "reason":
                result["reason"]
        })

    def create_dataframe(self):

        return pd.DataFrame(
            self.rows
        )

    def export_excel(
        self,
        output_dir
    ):

        df = self.create_dataframe()

        excel_path = os.path.join(
            output_dir,
            "seal_analysis.xlsx"
        )

        df.to_excel(
            excel_path,
            index=False,
            engine="openpyxl"
        )

        return excel_path

    def create_zip(
        self,
        output_dir,
        zip_name="predictions.zip"
    ):

        with zipfile.ZipFile(
            zip_name,
            "w"
        ) as zipf:

            for file_name in os.listdir(
                output_dir
            ):

                file_path = os.path.join(
                    output_dir,
                    file_name
                )

                zipf.write(
                    file_path,
                    arcname=file_name
                )

        return zip_name