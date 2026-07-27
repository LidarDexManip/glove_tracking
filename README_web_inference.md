# HOT3D browser inference UI

`hand_restoration_web.py` runs inference on the GPU server and sends only
images and metrics to the browser. It reads the formal derived train/holdout
manifests and discovers `controlnet_step*.pt` plus `controlnet_final.pt`.

## Install

```bash
cd /root/glove_tracking
conda run -n glove-hot3d python -m pip install -r requirements.web-ui.txt
```

## No SSH: temporary password-protected URL

Choose a strong password and keep the URL private:

```bash
export HAND_UI_USERNAME=glove
export HAND_UI_PASSWORD='REPLACE_WITH_A_LONG_RANDOM_PASSWORD'

systemd-run \
  --unit=hand-inference-web \
  --description="HOT3D browser inference" \
  --property=WorkingDirectory=/root/glove_tracking \
  --setenv=HAND_UI_USERNAME="$HAND_UI_USERNAME" \
  --setenv=HAND_UI_PASSWORD="$HAND_UI_PASSWORD" \
  --collect \
  /bin/bash -lc '
    exec env PYTHONUNBUFFERED=1 \
      HAND_UI_USERNAME="$HAND_UI_USERNAME" \
      HAND_UI_PASSWORD="$HAND_UI_PASSWORD" \
      CUDA_VISIBLE_DEVICES=0 \
      /root/miniconda3/bin/conda run --no-capture-output -n glove-hot3d \
      python hand_restoration_web.py --share \
      >> /root/glove_tracking/hand-inference-web.log 2>&1
  '
```

Read the generated URL:

```bash
tail -f /root/glove_tracking/hand-inference-web.log
```

If the environment variables are not preserved by the platform's
`systemd-run`, put their literal values inside the quoted service command or
use an environment file readable only by root.

## Platform port mapping

If the provider can expose a TCP port, avoid the temporary share tunnel:

```bash
export HAND_UI_USERNAME=glove
export HAND_UI_PASSWORD='REPLACE_WITH_A_LONG_RANDOM_PASSWORD'
python hand_restoration_web.py --host 0.0.0.0 --port 7860
```

Expose port 7860 through the provider console. Authentication is mandatory
whenever the UI is remotely reachable.

## Operations

```bash
systemctl status hand-inference-web
systemctl stop hand-inference-web
tail -f /root/glove_tracking/hand-inference-web.log
```

Generated experiments are saved under:

```text
outputs/hand_restoration/web_experiments/
```

The first inference request loads SD 1.5 and the selected ControlNet, so it is
slower than later requests. Selecting another checkpoint reloads only
ControlNet weights. Requests are serialized to protect GPU memory.
