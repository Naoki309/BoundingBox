import os
import json
from plyfile import PlyData

def convert_ply_to_json(ply_file_path, output_folder):
    """
    PLYファイルをJSON形式に変換する関数
    - ply_file_path: 変換対象のPLYファイルのフルパス
    - output_folder: 変換後のJSONファイルを保存するフォルダ
    """
    ply_file_name = os.path.basename(ply_file_path)
    json_file_name = ply_file_name.replace('.ply', '.json')
    json_file_path = os.path.join(output_folder, json_file_name)

    if os.path.exists(ply_file_path):
        try:
            # PLYファイルを読み込み（アスキーまたはバイナリ形式をサポート）
            ply_data = PlyData.read(ply_file_path)
            
            # 頂点データをリストに変換
            vertices = ply_data['vertex']
            vertex_list = []
            for vertex in vertices:
                vertex_list.append({
                    'x': float(vertex[0]),
                    'y': float(vertex[1]),
                    'z': float(vertex[2])
                })

            # 面データが存在する場合のみ取得
            face_list = []
            if 'face' in ply_data:
                faces = ply_data['face']
                for face in faces.data['vertex_indices']:
                    face_list.append([int(idx) for idx in face])

            # JSONデータとして保存するデータ
            bbox_data = {
                'vertices': vertex_list,
                'faces': face_list
            }

            # 出力フォルダが存在しない場合は作成
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # JSONファイルに書き込み
            with open(json_file_path, 'w') as json_file:
                json.dump(bbox_data, json_file, indent=4)

            print(f"PLYファイルをJSONに変換しました: {json_file_path}")
            return json_file_path
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return None
    else:
        print(f"指定されたPLYファイルが見つかりません: {ply_file_path}")
        return None
