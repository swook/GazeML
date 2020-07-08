import wget
import os

urls = ["https://ait.ethz.ch/projects/2018/landmarks-gaze/downloads/ELG_i180x108_f60x36_n64_m3.zip",
        "https://ait.ethz.ch/projects/2018/landmarks-gaze/downloads/ELG_i60x36_f60x36_n32_m2.zip"]


def get_weights(dest_dir):
    for url in urls:
        file_name = os.path.basename(url)
        dest_path = os.path.join(dest_dir, file_name)
        if os.path.exists(dest_path):
            print("Weight file {} already downloaded".format(file_name))
        else:
            print("Downloading weights from {} to {}".format(url, dest_path))
            os.makedirs(dest_dir, exist_ok=True)
            wget.download(url, out=dest_path)
