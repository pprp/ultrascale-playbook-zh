import requests
import json
import os


def upload_image_to_picgo(image_path):
    # PicGo Server 的默认地址和端口
    url = "http://127.0.0.1:36677/upload"

    # 检查图片路径是否存在
    if not os.path.exists(image_path):
        print(f"错误：图片路径 {image_path} 不存在。")
        return None

    # 构造请求体
    data = {"list": [image_path]}

    # 发送 POST 请求
    try:
        response = requests.post(url, json=data)
        response_data = response.json()
        if response_data.get("success"):
            print("上传成功！")
        else:
            print("上传失败！")
        print("返回数据：")
        print(json.dumps(response_data, indent=4, ensure_ascii=False))
        return response_data
    except requests.exceptions.RequestException as e:
        print(f"请求异常：{e}")
        return None
    except json.JSONDecodeError:
        print("解析响应数据失败")
        return None


# 示例用法
if __name__ == "__main__":
    image_path = input("请输入图片的本地路径：")
    upload_image_to_picgo(image_path)
