import os

def convert_obj_to_ply(obj_file_path, output_folder):
    """
    OBJファイルをPLY形式に変換する関数
    - obj_file_path: 変換対象のOBJファイルのフルパス
    - output_folder: 変換後のPLYファイルを保存するフォルダ
    """
    obj_file_name = os.path.basename(obj_file_path)
    ply_file_name = obj_file_name.replace('.obj', '.ply')
    ply_file_path = os.path.join(output_folder, ply_file_name)

    if os.path.exists(obj_file_path):
        try:
            vertices = []
            # OBJファイルを読み込み、頂点情報を抽出
            with open(obj_file_path, 'r') as obj_file:
                for line in obj_file:
                    if line.startswith('v '):  # 'v' で始まる行は頂点情報
                        parts = line.split()
                        vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                        vertices.append(vertex)

            # 出力フォルダが存在しない場合は作成
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # PLYファイルに書き込み
            with open(ply_file_path, 'w') as ply_file:
                # PLYファイルのヘッダー情報
                ply_file.write("ply\n")
                ply_file.write("format ascii 1.0\n")
                ply_file.write(f"element vertex {len(vertices)}\n")
                ply_file.write("property float x\n")
                ply_file.write("property float y\n")
                ply_file.write("property float z\n")
                ply_file.write("end_header\n")
                
                # 頂点情報を書き込み
                for vertex in vertices:
                    ply_file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
            
            print(f"OBJファイルをPLYに変換しました: {ply_file_path}")
            return ply_file_path
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return None
    else:
        print(f"指定されたOBJファイルが見つかりません: {obj_file_path}")
        return None