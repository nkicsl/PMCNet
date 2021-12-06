import numpy as np
import os
from keras import backend as K
from matplotlib import pyplot as plt
from keras.models import Model
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from lib import *
import tensorflow as tf
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
h=605
w=700
test_dir = "./data/test/images/"
test_label_dir = "./data/test/masks/" 
patch_size= 64
s= 64
save_dir='./result/'
images,labels=get_test_data(test_dir,test_label_dir,h,w)
new_images,new_h,new_w=paint_border_overlap(images,patch_size,s)
patches=extract_ordered_overlap(new_images, patch_size,s)
model_name='unet'   # unet  resunet denseunet unet_plus deeplabv3   pspnet   unet_refine  resunet_refine denseunet_refine deeplabv3_refine unetplus_refine
model_id=['23','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']
#model_id=['10','11','12','13','14','15']
#model_id=['16','17','18','19','20','21']
#model_id=['22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39']
#model_id=['28','29','30','31','32','33']
#model_id=['34','35','36','37','38','39']
#model_id=['40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57']
#model_id=['58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75']
#model_id=['22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75']

for k in range(55):
    print('\n'+'./new/'+model_name+model_id[k]+'.h5')
    model=load_model('./new/'+model_name+model_id[k]+'.h5',compile=False)
    predictions = model.predict(patches, batch_size=40, verbose=1)
    pred_patches = pred_to_imgs(predictions, patch_size, patch_size, "original") #N*patch_size*patch_size
    pred_imgs = recompone_overlap(pred_patches, new_h, new_w,s)
    pred_imgs = pred_imgs[:,0:h,0:w]
    for i in range(11):
        gray_image=images[i,:,:,1]
        label = labels[i]*255
        pred = pred_imgs[i]*255
        total_img = np.concatenate((gray_image,label,pred),axis=1)
        cv2.imwrite(save_dir+'Img_GT_Pre'+str(i)+'.jpg',total_img)

    y_scores, y_true = get_scores(pred_imgs,labels)
    #### AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    print("AUC_ROC:  " +str(AUC_ROC))
    roc =plt.figure()
    plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(save_dir+"ROC.png")
    plt.close() 

    ### PR
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  
    recall = np.fliplr([recall])[0] 
    AUC_prec_rec = np.trapz(precision,recall)
    print("AUC_PR: " +str(AUC_prec_rec))
    prec = plt.figure()
    plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('PR curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(save_dir+"PR.png")
    plt.close() 

    ####  Sen,Spe,Pre,F1-score
    threshold_confusion = 0.5
    y_pred = np.empty((y_scores.shape[0]))
    for i in range(y_scores.shape[0]):
        if y_scores[i]>=threshold_confusion:
            y_pred[i]=1
        else:
            y_pred[i]=0
    confusion = confusion_matrix(y_true, y_pred)


    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
    print("Global Accuracy: " +str(accuracy))

    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
    print("Specificity: " +str(specificity))

    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
    print("Sensitivity: " +str(sensitivity))

    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
    print("Precision: " +str(precision))

    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    print("F1 score: " +str(F1_score))

    file_perf = open(save_dir+'performances.txt', 'w')
    file_perf.write("AUC_ROC: "+str(AUC_ROC)
                    + "\nAUC_PR: " +str(AUC_prec_rec)
                    + "\nF1 score : " +str(F1_score)
                    +"\nAcc: " +str(accuracy)
                    +"\nSen: " +str(sensitivity)
                    +"\nSpe: " +str(specificity)
                    +"\nPre: " +str(precision))
    file_perf.close()