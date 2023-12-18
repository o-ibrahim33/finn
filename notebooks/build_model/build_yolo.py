import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
import shutil
import time

model_name = "lpyoloW4A4"
model_dir = os.environ['FINN_ROOT'] + "/notebooks"
model_file = model_dir + f"/{model_name}_quant.onnx"

estimates_output_dir = os.environ['FINN_ROOT']+f"/notebooks/output_finn/{model_name}"

#Delete previous run results if exist
if os.path.exists(estimates_output_dir):
    shutil.rmtree(estimates_output_dir)
    print("Previous run results deleted!")


cfg_estimates = build.DataflowBuildConfig(
    output_dir          = estimates_output_dir,
    mvau_wwidth_max     = 80,
    target_fps          = 300,
    synth_clk_period_ns = 10.0,
    board               = "U280",
    fpga_part           = "xcu280-fsvh2892-2L-e",
    steps               = build_cfg.yolo_build_steps,
    shell_flow_type     = build_cfg.ShellFlowType.VITIS_ALVEO,
    auto_fifo_depths    = True,
    force_rtl_conv_inp_gen = True,
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        build_cfg.DataflowOutputType.OOC_SYNTH,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE
    ]
)

start_time = time.time()
build.build_dataflow_cfg(model_file, cfg_estimates)
end_time = time.time()

duration = end_time - start_time

print(f"Operation took {duration:.6f} seconds")

