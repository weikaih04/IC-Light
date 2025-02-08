#!/bin/bash

TOTAL_JOBS=128

for (( JOB_INDEX=0; JOB_INDEX<TOTAL_JOBS; JOB_INDEX++ )); do
  echo "Submitting job ${JOB_INDEX} out of $((TOTAL_JOBS-1))"
  
  # Export variables so that envsubst can replace the placeholders in the template.
  export TOTAL_JOBS
  export JOB_INDEX

  # Substitute variables in the YAML template and save to a temporary YAML file.
  envsubst < job_template.yaml > job_${JOB_INDEX}.yaml
  
  # Submit the job with gantry using a unique job name.
  gantry run --allow-dirty --name "relightening_${JOB_INDEX}_$((TOTAL_JOBS - 1))_2_3_v2" job_${JOB_INDEX}.yaml
done

echo "All ${TOTAL_JOBS} jobs have been submitted."