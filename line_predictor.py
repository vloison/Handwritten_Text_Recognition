from params import *
from network import *
from data.Preprocessing import img_resize

import os
import torch as torch
from tqdm import tqdm
from torch.autograd import Variable
from skimage import io as img_io


params = BaseOptions().parser()


def predict(model, data_loc, imgH, imgW):
    # Create folder to stock predictions
    pred_path = data_loc + 'predictions.txt'
    # If file does not exist, create it . THen write/overwrite in it
    if not os.path.exists(pred_path):
        label_file = open(pred_path, 'x')
        label_file.close()
    with open(pred_path, 'w') as label_file:
        label_file.write("Image label -- Prediction \n \n")

    print("Starting predictions...")
    model.eval()

    # Go through data folder to make predictions
    for filename in tqdm(os.listdir(data_loc)):
        # print(filename)
        # Process predictions
        if filename.endswith(".jpg"):
            img = img_io.imread(data_loc + filename, plugin='matplotlib', as_gray=True)
            img = img_resize(img, height=imgH, width=imgW, keep_ratio=True)
            img = torch.Tensor(img).float().unsqueeze(0)
            img = Variable(img.unsqueeze(1))
            # print('img shape', img.shape)
            if params.cuda and torch.cuda.is_available():
                img = img.cuda()
            # print(img.type)
            with torch.no_grad():
                pred = model(img)
            pred_size = Variable(torch.LongTensor([pred.size(0)] * img.size(0)))

            # Convert probability output to string
            tdec = pred.argmax(2).permute(1, 0).cpu().numpy().squeeze()
            # print(tdec)
            # print(tdec.ndim)
            # Convert path to label, batch has size 1 here
            if tdec.ndim == 0:
                dec_transcr = ''.join([params.icdict[tdec.item()]]).replace('_', '')
            else:
                tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
                dec_transcr = ''.join([params.icdict[t] for t in tt]).replace('_', '')

            # Write lavel in file

            # Save label file
            with open(pred_path, 'a') as label_file:
                label_file.write(filename + "   --  " + dec_transcr + "\n")

    print("Predictions done and saved at location " + pred_path)


if __name__ == "__main__":

    MODEL = RCNN(imheight=params.imgH,
                        nc=params.NC,
                        n_conv_layers=params.N_CONV_LAYERS,
                        n_conv_out=params.N_CONV_OUT,
                        conv=params.CONV,
                        batch_norm=params.BATCH_NORM,
                        max_pool=params.MAX_POOL,
                        n_r_layers=params.N_REC_LAYERS,
                        n_r_input=params.N_REC_INPUT,
                        n_hidden=params.N_HIDDEN,
                        n_out=len(params.alphabet),
                        bidirectional=params.BIDIRECTIONAL,
                        feat_extractor=params.feat_extractor,
                        dropout=params.DROPOUT)
    #
    # MODEL.load_state_dict(torch.load('/media/vn_nguyen/hdd/hux/Results/08-19_12:48:18/IAM_model_imgH64.pth'))
    # print(MODEL)
    # torch.save(MODEL, '/home/loisonv/Text_Recognition/trained_networks/ICFHR2014_model_imgH32.pth')

    # MODEL = torch.load('/home/hux/HTR/trained_networks/IAM_model_imgH64.pth')
    MODEL.load_state_dict(torch.load(params.model_path))

    if params.cuda and torch.cuda.is_available():
        MODEL = MODEL.cuda()
    DATA_LOC = params.data_path

    predict(MODEL, DATA_LOC, imgH=params.imgH, imgW=params.imgW)

