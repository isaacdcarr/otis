from keras.preprocessing import image
import os

if __name__ == '__main__':
    w = []
    h = []

    img_dir = 'data/chest_xray/train/NORMAL/'
    for file in os.listdir(img_dir):
        if 'jpeg' in file:
            img = image.load_img(img_dir + file)
            w.append(float(img.size[0]))
            h.append(float(img.size[1]))


    output = open('src/preprocessing/checksize.txt','w')
    output.write("... summary ...\n")
    output.write("Average width:\t" + str(sum(w)/len(w)) + "\n")
    output.write("Average height:\t" + str(sum(h)/len(h)) + "\n")
    output.write("Max width:\t" + str(max(w)) + "\n")
    output.write("Max height:\t" + str(max(h)) + "\n")
    output.write("Min width:\t" + str(min(w)) + "\n")
    output.write("Min height:\t" + str(min(h)) + "\n")
    output.close()