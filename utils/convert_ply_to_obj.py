import os
import trimesh

def convert_ply_to_obj(ply_file_path, output_folder):
    """
    PLYファイルをOBJファイルに変換する関数
    - ply_file_path: 変換対象のPLYファイルのフルパス
    - output_folder: 変換後のOBJファイルを保存するフォルダ
    """
    ply_file_name = os.path.basename(ply_file_path)
    obj_file_name = ply_file_name.replace('.ply', '.obj')
    obj_file_path = os.path.join(output_folder, obj_file_name)

    if os.path.exists(ply_file_path):
        try:
            # PLYファイルを読み込み
            mesh = trimesh.load(ply_file_path)

            # 出力フォルダが存在しない場合は作成
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # OBJファイルにエクスポート
            mesh.export(obj_file_path, file_type='obj')
            print(f"PLYファイルをOBJに変換しました: {obj_file_path}")
            return obj_file_path
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return None
    else:
        print(f"指定されたPLYファイルが見つかりません: {ply_file_path}")
        return None