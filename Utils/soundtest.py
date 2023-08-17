import os
import sys

def check_os():
    if sys.platform.startswith('linux'):
        return "linux"
    elif sys.platform.startswith('win'):
        return "windows"
    elif sys.platform.startswith('darwin'):
        return "mac"
    else:
        return None

def list_audio_devices():
    os_type = check_os()
    if os_type == "linux":
        os.system("aplay -l")
    # 他のOSに関してのコードもここに追加可能

def play_test_sound():
    os_type = check_os()
    if os_type == "linux":
        os.system("aplay /usr/share/sounds/alsa/Front_Center.wav")
    # 他のOSに関してのコードもここに追加可能

if __name__ == "__main__":
    print("### OS確認 ###")
    os_type = check_os()
    if not os_type:
        print("未対応のOSです。")
        sys.exit(1)
    else:
        print(f"現在のOS: {os_type}")

    print("\n### オーディオデバイス一覧 ###")
    list_audio_devices()

    input("\nヘッドフォンを接続して、Enterを押してください。")
    
    print("\n### テスト音再生 ###")
    play_test_sound()
    
    input("ヘッドフォンから音が聞こえたら、スクリプトを終了してください。")
