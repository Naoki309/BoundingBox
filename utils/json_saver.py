import json

def save_bounding_boxes_as_json(batch_pred_map_cls, json_file_path='bounding_boxes_results.json'):
    """
    バウンディングボックスの情報をJSONファイルに保存する
    Args:
        batch_pred_map_cls: バッチごとにバウンディングボックスの情報を保持したリスト
        json_file_path: 保存するJSONファイルのパス
    """
    # バウンディングボックス情報をJSONに保存
    with open(json_file_path, 'w') as json_file:
        json.dump(batch_pred_map_cls, json_file, indent=4)
    
    print(f"Bounding box results saved to {json_file_path}")
