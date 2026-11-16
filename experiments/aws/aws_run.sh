#!/usr/bin/env bash
# Run the Colab notebook on an EC2 GPU instead of Colab.
#   up      check HF access, launch (L4 -> A10G -> T4 fallback), upload, start
#   rerun   rebuild + re-upload notebook after a code fix, restart (finished items are skipped)
#   status  tail the run log
#   watch   block until the run finishes or a step fails (for the babysitting loop)
#   fetch   download executed notebook + results zip, stop the box
# Secrets file ~/.verascope_secrets (KEY=value): HF_TOKEN, GROQ_API_KEY, optional PYANNOTEAI_API_KEY.
set -euo pipefail
export AWS_PROFILE=${AWS_PROFILE:-verascope} AWS_DEFAULT_REGION=us-east-1
HERE=$(cd "$(dirname "$0")" && pwd); NB=$HERE/../notebooks
SECRETS=${SECRETS:-$HOME/.verascope_secrets}
SG=sg-0c3e341159d2fac85      # verascope-sg (specs/docs/aws-gpu.md)
KEY=$HOME/.ssh/verascope.pem
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
STATE=$HERE/.instance        # instance id of the current run

host() { aws ec2 describe-instances --instance-ids "$(cat "$STATE")" \
           --query 'Reservations[0].Instances[0].PublicDnsName' --output text; }
upload() {  # $1 = host
  python3 "$HERE/../build_notebook.py"
  scp -q -i "$KEY" -r "$HERE/shim" "$HERE/box.sh" ubuntu@$1:~
  scp -q -i "$KEY" "$NB/verascope_experiments.ipynb" ubuntu@$1:~/nb.ipynb
  $SSH ubuntu@$1 'pkill -f "[p]apermill"; pkill -f "bash /home/ubuntu/[b]ox.sh"; sleep 2; nohup bash ~/box.sh >> ~/box.log 2>&1 &'
}

case ${1:-} in
check)  # HTTP codes only, never values; 200 = OK
  set -a; . "$SECRETS"; set +a
  for r in speaker-diarization-3.1 segmentation-3.0; do
    echo "HF pyannote/$r: $(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $HF_TOKEN" \
          https://huggingface.co/pyannote/$r/resolve/main/config.yaml)"
  done
  [ -n "${GROQ_API_KEY:-}" ] && echo "Groq: $(curl -s -o /dev/null -w '%{http_code}' \
      -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/openai/v1/models)"
  [ -n "${PYANNOTEAI_API_KEY:-}" ] && echo "pyannoteAI: $(curl -s -o /dev/null -w '%{http_code}' \
      -H "Authorization: Bearer $PYANNOTEAI_API_KEY" https://api.pyannote.ai/v1/test)"
  aws sts get-caller-identity --query Arn --output text ;;
up)
  [ -s "$SECRETS" ] || { echo "missing $SECRETS"; exit 1; }
  TOK=$(grep '^HF_TOKEN=' "$SECRETS" | cut -d= -f2- | tr -d " \"'")
  for r in speaker-diarization-3.1 segmentation-3.0; do   # the Colab 401; fail here, not 1 h into the run
    c=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOK" \
        "https://huggingface.co/pyannote/$r/resolve/main/config.yaml")
    [ "$c" = 200 ] || { echo "HF_TOKEN cannot read pyannote/$r (HTTP $c)"; exit 1; }
  done
  AMI=$(aws ssm get-parameter --query Parameter.Value --output text --name \
        /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id)
  aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 22 \
      --cidr "$(curl -s https://checkip.amazonaws.com)/32" >/dev/null 2>&1 || true
  IID=
  for t in ${TYPES:-g6.xlarge g5.xlarge g4dn.xlarge}; do
    IID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$t" --key-name verascope \
      --security-group-ids $SG --instance-initiated-shutdown-behavior stop \
      --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":150,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=verascope-experiments}]' \
      --query 'Instances[0].InstanceId' --output text) && break || echo "$t unavailable, trying next"
  done
  [ -n "$IID" ] || exit 1
  echo "$IID" > "$STATE"; echo "instance $IID ($t)"
  aws ec2 wait instance-running --instance-ids "$IID"
  H=$(host); until $SSH ubuntu@$H true 2>/dev/null; do sleep 10; done
  scp -q -i "$KEY" "$SECRETS" ubuntu@$H:~/secrets.env
  $SSH ubuntu@$H 'chmod 600 ~/secrets.env'
  upload "$H"; echo "started on $H" ;;
rerun)
  H=$(host); upload "$H"; echo "restarted on $H" ;;
status)
  $SSH ubuntu@"$(host)" "cat ~/DONE 2>/dev/null; tail -n ${N:-25} ~/box.log" ;;
watch)
  # Exits when the run is done, or a step fails that wasn't in the log when watch started.
  H=$(host); base=$($SSH ubuntu@$H "grep -cE '==> .*exit [1-9]|access: FAILED|PapermillExecutionError|^ERROR' ~/box.log; true")
  while sleep 120; do
    s=$($SSH ubuntu@$H "cat ~/DONE 2>/dev/null; grep -cE '==> .*exit [1-9]|access: FAILED|PapermillExecutionError|^ERROR' ~/box.log; true" 2>/dev/null) || continue
    case $s in papermill*) echo "$s"; exit 0 ;; esac
    [ "$s" -gt "$base" ] && { echo "new failure"; exit 3; }
  done ;;
fetch)
  H=$(host)
  scp -i "$KEY" ubuntu@$H:~/out.ipynb "$NB/verascope_experiments_executed.ipynb"
  scp -i "$KEY" ubuntu@$H:~/box.log "$NB/aws_box.log"
  scp -i "$KEY" "ubuntu@$H:/content/drive/MyDrive/verascope/verascope_results.zip" "$NB/" || true
  aws ec2 stop-instances --instance-ids "$(cat "$STATE")" >/dev/null
  echo "fetched to $NB; box stopping" ;;
*) sed -n 2,8p "$0"; exit 1 ;;
esac
