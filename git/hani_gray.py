from asyncio import shield
from time import time
import cv2
from cv2 import destroyAllWindows

def frame_resize(frame,n=2):
    return cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))

if __name__ == "__main__":

    tracker = cv2.TrackerKCF_create()

    cap = cv2.VideoCapture(0)

    while True:
        ret,frame = cap.read()
        if not ret:
            continue
        frame = frame_resize(frame)
        bbox = (0,0,10,10)
        bbox= cv2.selectROI(frame,False)
        ok = tracker.init(frame,bbox)
        cv2,destroyAllWindows()
        break

    while True:
        ret,frame = cap.read()
        frame = frame_resize(frame)
        if not ret:
            k = cv2.waitKey(1)
            if k == 27:
                break
            continue

        timer = cv2.getTickCount()

        track,bbox = tracker.update(frame)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        
        if track:
            p1 = (int(bbox[0]),int(bbox[1]))
            p2 = (int(bbox[0]+bbox[2]),int(bbox[1] + bbox[3]))
            cv2.rectangle(frame,p1,p2,(0,0,0),1,1)

        else:
            cv2.putText(frame,"Failuer",(10,50),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA);
        
        cv2.putText(frame, "FPS : " + str(int(fps)), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);

        # 加工済の画像を表示する
        cv2.imshow("Tracking", frame)

        # キー入力を1ms待って、k が27（ESC）だったらBreakする
        k = cv2.waitKey(1)
        if k == 27 :
            break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()

