import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import json

class Objhandler(FileSystemEventHandler):
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(".obj"):
            print(f"新しいOBJファイルが検出されました: {event.src_path}")
            self.process_obj(event.src_path)

    def process_obj(self, obj_path):
        try:
            # OBJからPLYへの変換
            ply_path = self.convert_obj_to_ply(obj_path)

            # Votenetの実行
            bbox_info = self.run_votenet(ply_path)

            # BBox情報をOBJとして保存
            bbox_obj_path = self.save_bbox_as_obj(bbox_info, obj_path)

            print(f"処理完了: {bbox_obj_path}")

        except Exception as e:
            print(f"処理中にエラーが発生しました: {e}")

    def convert_obj_to_ply(self, obj_path):
        ply_path = obj_path.replace(".obj", ".ply")
        cmd = f"meshlabserver -i \"{obj_path}\" -o \"{ply_path}\""
        subprocess.run(cmd, shell=True, check=True)
        print(f"OBJをPLYに変換しました: {ply_path}")
        return ply_path

    def run_votenet(self, ply_path):
        # Votenetの実行コマンドを実装
        output_json_path = ply_path.replace(".ply", "_bbox.json")
        cmd = f"python votenet_script.py --input \"{ply_path}\" --output \"{output_json_path}\""
        subprocess.run(cmd, shell=True, check=True)
        print(f"Votenetを実行しました: {ply_path} -> {output_json_path}")

        # JSONファイルの読み込み
        with open(output_json_path, 'r') as f:
            bbox_info = json.load(f)

        return bbox_info

    def save_bbox_as_obj(self, bbox_info, original_obj_path):
        bbox_obj_path = original_obj_path.replace(".obj", "_bbox.obj")
        with open(bbox_obj_path, 'w') as f:
            f.write("# Bounding Boxes\n")
            for bbox in bbox_info.get('bboxes', []):
                # 各BBoxの頂点をOBJ形式で記述
                # bboxにはx_min, y_min, z_min, x_max, y_max, z_maxが含まれると仮定
                x_min, y_min, z_min = bbox['x_min'], bbox['y_min'], bbox['z_min']
                x_max, y_max, z_max = bbox['x_max'], bbox['y_max'], bbox['z_max']
                vertices = [
                    (x_min, y_min, z_min),
                    (x_max, y_min, z_min),
                    (x_max, y_max, z_min),
                    (x_min, y_max, z_min),
                    (x_min, y_min, z_max),
                    (x_max, y_min, z_max),
                    (x_max, y_max, z_max),
                    (x_min, y_max, z_max),
                ]
                vertex_offset = 1  # 現在の頂点数に基づいてオフセットを計算する必要あり
                for v in vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                # 面の定義（四角形を二つの三角形に分割）
                f.write(f"f {vertex_offset} {vertex_offset+1} {vertex_offset+2}\n")
                f.write(f"f {vertex_offset} {vertex_offset+2} {vertex_offset+3}\n")
        print(f"BBox情報をOBJとして保存しました: {bbox_obj_path}")
        return bbox_obj_path

if __name__ == "__main__":
    upload_folder = "/mnt/c/SharedFolder/"
    event_handler = Objhandler(upload_folder)
    observer = Observer()
    observer.schedule(event_handler, path=upload_folder, recursive=False)
    observer.start()
    print(f"共有フォルダを監視しています: {upload_folder}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()