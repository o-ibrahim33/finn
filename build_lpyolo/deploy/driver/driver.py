
# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import os
import sys
from qonnx.core.datatype import DataType
from driver_base import FINNExampleOverlay
from pynq.pl_server.device import Device


# Add the parent directory to the sys.path list
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../..'))
print(parent_dir)
sys.path.append(parent_dir)

from utils import *
from detect import Detect


# dictionary describing the I/O of the FINN-generated accelerator
io_shape_dict = {
    # FINN DataType for input and output tensors
    "idt" : [DataType['UINT8']],
    "odt" : [DataType['INT17']],
    # shapes for input and output tensors (NHWC layout)
    "ishape_normal" : [(1, 384, 640, 3)],
    "oshape_normal" : [(1, 12, 20, 36)],
    # folded / packed shapes below depend on idt/odt and input/output
    # PE/SIMD parallelization settings -- these are calculated by the
    # FINN compiler.
    "ishape_folded" : [(1, 384, 640, 1, 3)],
    "oshape_folded" : [(1, 12, 20, 36, 1)],
    "ishape_packed" : [(1, 384, 640, 1, 3)],
    "oshape_packed" : [(1, 12, 20, 36, 3)],
    "input_dma_name" : ['idma0'],
    "output_dma_name" : ['odma0'],
    "number_of_external_weights": 0,
    "num_inputs" : 1,
    "num_outputs" : 1,
}

def execute(image, image_org, image_name, save=True):
    accel_start_time = time.time()
    accel_out = accel.execute(image)
    accel_out = accel_out.transpose(0,3,1,2)
    accel_out = accel_out*scale
    accel_out = torch.from_numpy(accel_out)
    accel_elapsed_time = time.time() - accel_start_time

    head_start_time = time.time()
    head_out = detect_head([accel_out,accel_out,accel_out])[0]
    head_out = head_out.detach().numpy()
    head_out = np.reshape(head_out,(1, 2160, 12))
    head_out = torch.from_numpy(head_out)
    head_elapsed_time = time.time() - head_start_time

    pred = non_max_suppression(head_out,conf_thres = 0.20, iou_thres=0.35)
    image_boxes = plot_boxes(pred, image[0], image_org)

    if save:
        cv2.imwrite(output_dir+f"/{image_name}",image_boxes)
    return image_boxes, accel_elapsed_time, head_elapsed_time

def execute_batch(batch, image_org, image_name, save=True):
    accel_start_time = time.time()
    accel_out = accel.execute(batch)
    accel_out = accel_out.transpose(0,3,1,2)
    accel_out = accel_out*scale
    accel_out = torch.from_numpy(accel_out)
    accel_elapsed_time = time.time() - accel_start_time
    print(accel_out.shape,accel_elapsed_time)

    head_start_time = time.time()
    head_out = detect_head([accel_out,accel_out,accel_out])[0]
    head_out = head_out.detach().numpy()
    head_out = np.reshape(head_out,(1, 2160, 12))
    head_out = torch.from_numpy(head_out)
    head_elapsed_time = time.time() - head_start_time

    pred = non_max_suppression(head_out,conf_thres = 0.20, iou_thres=0.35)
    #image_boxes = plot_boxes(pred, image, image_org)

    if save:
        cv2.imwrite(output_dir+f"/{image_name}",image_boxes)
    return image_boxes, accel_elapsed_time, head_elapsed_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute FINN-generated accelerator on numpy inputs, or run throughput test')
    parser.add_argument('--exec_mode', help='Please select functional verification ("execute") or throughput test ("throughput_test")', default="execute")
    parser.add_argument('--platform', help='Target platform: zynq-iodma alveo', default="alveo")
    parser.add_argument('--batchsize', help='number of samples for inference', type=int, default=1)
    parser.add_argument('--device', help='FPGA device to be used', type=int, default=0)
    parser.add_argument('--bitfile', help='name of bitfile (i.e. "resizer.bit")', default="resizer.bit")
    parser.add_argument('--inputfile', help='name(s) of input npy file(s) (i.e. "input.npy")', nargs="*", type=str, default=["input.npy"])
    parser.add_argument('--outputfile', help='name(s) of output npy file(s) (i.e. "output.npy")', nargs="*", type=str, default=["output.npy"])
    parser.add_argument('--runtime_weight_dir', help='', default="runtime_weights/")
    parser.add_argument('--dataset_dir', help='', default="")
    parser.add_argument('--output_dir', help='', default="../output")
    # parse arguments
    args = parser.parse_args()
    exec_mode = args.exec_mode
    platform = args.platform
    batch_size = args.batchsize
    bitfile = args.bitfile
    inputfile = args.inputfile
    outputfile = args.outputfile
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    runtime_weight_dir = args.runtime_weight_dir
    devID = args.device
    device = Device.devices[devID]

    # instantiate FINN accelerator driver and pass batchsize and bitfile
    accel = FINNExampleOverlay(
        bitfile_name = bitfile, platform = platform,
        io_shape_dict = io_shape_dict, batch_size = batch_size,
        runtime_weight_dir = runtime_weight_dir, device=device
    )

    nc = 7
    anchors = np.array([[10, 13, 16, 30, 33, 23], [81, 82, 135, 169, 344, 319], [116, 90, 156, 198, 373, 326]])
    detect_head = Detect(nc, anchors,ch=[36,36,36],do_quant=False)
    detect_head.load_state_dict(torch.load("../models/detect_module.pt",map_location=torch.device('cpu')))
    detect_head.eval()

    scale = np.load("scale.npy")
    # for the remote execution the data from the input npy file has to be loaded,
    # packed and copied to the PYNQ buffer
    if exec_mode == "execute":
        # load desired input .npy file(s)
        ibuf_normal = []
        for ifn in inputfile:
            ibuf_normal.append(np.load(ifn))
        obuf_normal = accel.execute(ibuf_normal)
        if not isinstance(obuf_normal, list):
            obuf_normal = [obuf_normal]
        for o, obuf in enumerate(obuf_normal):
            np.save(outputfile[o], obuf)
    elif exec_mode == "execute_dataset":
        images_name, images, images_org = load_and_preprocess_dir(dataset_dir)
        
        # for i in range(0, len(images), batch_size):
        #     batch_images = images[i:i + batch_size]
        #     batch_names = images_name[i:i + batch_size]
        #     batch_images_org = images_org[i:i + batch_size]
            
        #     execute_batch(batch_images, batch_names, batch_images_org)
        total_accel_elapsed_time = 0
        runtime = 0
        for idx, image in enumerate(images):
            _, accel_elapsed_time,head_elapsed_time = \
            execute(
                image,
                images_org[idx],
                images_name[idx]
            )
            total_accel_elapsed_time += accel_elapsed_time
            runtime += accel_elapsed_time + head_elapsed_time
        print(f"throughput_accel[images/s] {len(images)/total_accel_elapsed_time}, \
                throughput_total[images/s] {len(images)/runtime}")

    elif exec_mode == "execute_video":
        video_path = '../data/street.mp4'
        vid = cv2.VideoCapture(video_path) 

        if not vid.isOpened():
            print("Error: Could not open camera.")
            exit()

        while(True): 
            ret, frame = vid.read()

            frame_org = frame.copy()
            frame = preprocess_image(frame)

            image_boxes = execute(frame, frame_org, save=False) 
        
            # Display the resulting frame 
            cv2.imshow('frame', image_boxes) 
            
            # the 'q' button is set as the 
            # quitting button
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        
        # After the loop release the cap object 
        vid.release() 
        # Destroy all the windows 
        cv2.destroyAllWindows() 
    elif exec_mode == "throughput_test":
        # remove old metrics file
        try:
            os.remove("nw_metrics.txt")
        except FileNotFoundError:
            pass
        res = accel.throughput_test()
        file = open("nw_metrics.txt", "w")
        file.write(str(res))
        file.close()
        print("Results written to nw_metrics.txt")
    else:
        raise Exception("Exec mode has to be set to execute or throughput_test")
