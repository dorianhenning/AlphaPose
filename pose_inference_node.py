# ROS Node for fast pose inference using pytorch implementations of YOLOv3 & SPPE
#
# Author: Dorian Henning
# Date: 25/07/2019
# Contact: dorian.henning@gmail.com
#
# Copyright: Imperial College London
#
# Based on the AlphaPose Framework and used/distributed under the original LICENSE

from opt import opt

from dataloader import ImageLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from SPPE.src.main_fast_inference import *

import os
import sys
from tqdm import tqdm
import time
from fn import getTime

from pPose_nms import pose_nms, write_json

args = opt
args.dataset = 'coco'
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    inputpath = args.inputpath
    inputlist = args.inputlist
    mode = args.mode
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    # Receive input images
    data_loader = ImageLoader(im_names, batchSize=args.detbatch, format='yolo').start()

    # ROS Node Initialization
    # Do all loading of models etc. first

    # Load detection loader
    print("Loading YOLO model..")
    sys.stdout.flush()
    det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start()
    det_processor = DetectionProcessor(det_loader).start()

    # Load pose model
    pose_dataset = Mscoco()
    pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Init data writer
    writer = DataWriter(args.save_video).start()

    data_len = data_loader.length()
    im_names_desc = tqdm(range(data_len))

    batchSize = 1
    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                continue

            ckpt_time, det_time = getTime(start_time)
            runtime_profile['dt'].append(det_time)
            # Pose Estimation

            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)
            hm = hm.cpu()
            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])

            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)

        if args.profile:
            # TQDM
            im_names_desc.set_description(
                "det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}".format(
                    dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']),
                    pn=np.mean(runtime_profile['pn']))
            )

    print("===========================> Finish Model Running.")
    if (args.save_img or args.save_video) and not args.vis_fast:
        print("===========================> Rendering remaining images in the queue...")
        print("===========================> If this step takes too long, you can enable "
              "the --vis_fast flag to use fast rendering (real-time).")
    while (writer.running()):
        pass
    writer.stop()
    final_result = writer.results()
    write_json(final_result, args.outputpath)
