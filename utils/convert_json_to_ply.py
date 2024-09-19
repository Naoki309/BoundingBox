import os
import json
import struct

def convert_json_to_ply(json_file_path, output_folder):
    """
    JSONファイルをバイナリ形式のPLYファイルに変換する関数
    - json_file_path: Unityから送られたJSONファイルのフルパス
    - output_folder: PLYファイルを保存するフォルダ
    """
    json_file_name = os.path.basename(json_file_path)  # JSONファイルの名前
    ply_file_name = json_file_name.replace('.json', '.ply')  # PLYファイル名に変更
    ply_file_path = os.path.join(output_folder, ply_file_name)  # 保存先パス

    if os.path.exists(json_file_path):
        try:
            # JSONファイルの読み込み
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)

            # 'vertices' が含まれているか確認
            if 'vertices' not in data:
                raise ValueError("JSONデータに'vertices'フィールドが見つかりません")

            vertices = data['vertices']  # 頂点データを取得
            faces = data.get('faces', [])  # 面データを取得

            # 出力フォルダが存在しない場合は作成
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # PLYファイルに書き込み
            with open(ply_file_path, 'wb') as ply_file:
                # PLYファイルのヘッダー情報
                header = (
                    "ply\n"
                    "format binary_little_endian 1.0\n"
                    f"element vertex {len(vertices)}\n"
                    "property float x\n"
                    "property float y\n"
                    "property float z\n"
                    "property uchar red\n"
                    "property uchar green\n"
                    "property uchar blue\n"
                    f"element face {len(faces)}\n"
                    "property list uchar int vertex_indices\n"
                    "end_header\n"
                )
                ply_file.write(header.encode('ascii'))

                # 頂点データをバイナリ形式で書き込み
                for vertex in vertices:
                    ply_file.write(struct.pack('<fff', vertex['x'], vertex['y'], vertex['z']))
                    # 頂点の色情報がなければ白で埋める
                    ply_file.write(struct.pack('BBB', 255, 255, 255))

                # 面データを書き込み
                for face in faces:
                    if len(face) == 3:  # 三角形の面データ
                        ply_file.write(struct.pack('Biii', len(face), face[0], face[1], face[2]))

            print(f"JSONファイルをPLYファイルに変換しました: {ply_file_path}")
            return ply_file_path
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return None
    else:
        print(f"指定されたJSONファイルが見つかりません: {json_file_path}")
        return None
