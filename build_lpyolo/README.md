# Build LPYOLO using FINN and executing it

## Build LPYOLO
To build lpyolo, you need to have the onnx model under `build_lpyolo/models` in the format of
`{MODEL_NAME}_quant.onnx`. 

Building the model consists of several steps specific to each model to do the transformations needed to transform a brevitas model to a FINN accelator.
running `build_lpyolo.py` has to be done from the docker container.

Run this command under the root directoy of finn 

```sh
./run-docker.sh
```

After successfully running the docker image run these commands

```sh
cd build_lpyolo
```
```sh
python3 build_lpyolo.py --onnx_model_name MODEL_NAME
```
where the default MODEL_NAME is lpyoloW4A4 which is lpyolo with 4 bit weight quantization and activation

the output should be like this

```sh
Running step: step_yolo_tidy [1/13]
Running step: step_yolo_streamline [2/13]
Running step: step_yolo_convert_to_hls [3/13]
Running step: step_create_dataflow_partition [4/13]
Running step: step_target_fps_parallelization [5/13]
Running step: step_generate_estimate_reports [6/13]
Running step: step_hls_codegen [7/13]
Running step: step_hls_ipgen [8/13]
```

The output files which include the drivers, weights and reports will be generated under
`build_lpyolo/output_finn/MODEL_NAME`

## Running LPYOLO Accelator

**Note** : The generated driver after running `build_lpyolo.py` does not include the detect head, it needs to be added manually, which is already done in `build_lpyolo/deploy`

Also `scale.npy` which is the dequantization scale which should be multiplied by the accelelator output before feeding it to the detect head can be find in the **Netron** graph of the model.

Each model has a different `scale.npy`, so for each model this .npy file has to be downloaded manually and copied to `build_lpyolo/deploy/`, this is already done for model 4WA4.

Before running the driver, some commads needs to be run beforehand.

**Run it locally not from the docker container**

```sh
sudo -E -s
source /opt/xilinx/xrt/setup.sh
```

And go to the deploy directory 
```sh
cd build_lpyolo/deploy
```

### Run in throughput_test mode
```sh
python3 driver/driver.py --bitfile bitfile/finn-accel.xclbin --runtime_weight_dir driver/runtime_weights --exec_mode throughput_test
```
### Run in execute mode
```sh
python3 driver/driver.py --bitfile bitfile/finn-accel.xclbin --runtime_weight_dir driver/runtime_weights --exec_mode execute --inputfile /path/to/input.npy --outputfile /path/to/output.npy
```
### Run in execute_dataset mode
```sh
python3 driver/driver.py --bitfile bitfile/finn-accel.xclbin --runtime_weight_dir driver/runtime_weights --exec_mode execute_dataset --dataset_dir /path/to/dataset --output_dir /path/to/output_dir
```



