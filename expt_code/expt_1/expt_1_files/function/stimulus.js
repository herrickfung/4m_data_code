////////////////////////////////////////////////////////////////////////////////
// these functions creates stimulus

function createFixation(canvas) {
  var window_dims = getScreenDims();
  var ctx = canvas.getContext('2d');
  ctx.strokeStyle = 'white';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(window_dims[1]/2-10, window_dims[0]/2);
  ctx.lineTo(window_dims[1]/2+10, window_dims[0]/2);
  ctx.moveTo(window_dims[1]/2, window_dims[0]/2-10);
  ctx.lineTo(window_dims[1]/2, window_dims[0]/2+10);
  ctx.stroke();
};

function createGabor(canvas){
  var ctx = canvas.getContext('2d');
  var window_dims = getScreenDims();
  var triallist_no = getCurrentTriallist();
  var trial = getCurrentTrialNo();
  var stimSize = triallist[triallist_no][trial][0];
  var contrast = triallist[triallist_no][trial][2];
  var SD_mask = stimSize/8;
  var trial_tilt = computeTrialTilt(triallist[triallist_no][trial][3], triallist[triallist_no][trial][4], triallist[triallist_no][trial][5])
  var numCyclesPer_stimSize = 8;   // for gabor stimulus, number of cycels per stimulus size

  var arr = jglMakeGrating(stimSize, stimSize, numCyclesPer_stimSize, trial_tilt, 0);
  var mask = make2dMask(arr, 1, SD_mask);
  var gabor = jglCreateGabor(ctx, arr, mask, contrast);
  var offset = (window_dims[0] - window_dims[1])/2
  ctx.putImageData(gabor, (window_dims[0] - stimSize)/2 - offset , (window_dims[1] - stimSize)/2 + offset); //display in the center

};

function createGaborStart(trial){
  var triallist_no = getCurrentTriallist();
  var current_trial_no = getCurrentTrialNo();
  trial.stimulus_duration = triallist[triallist_no][current_trial_no][1];
  trial.trial_duration = triallist[triallist_no][current_trial_no][1] + 100;  // add 100ms for iti
};

function createGaborEnd(data){
  var triallist_no = getCurrentTriallist();
  var trial = getCurrentTrialNo();
  data.stim_size = triallist[triallist_no][trial][0];
  data.stim_dur = triallist[triallist_no][trial][1];
  data.stim_contrast = triallist[triallist_no][trial][2];
  data.stim_tilt_diff = triallist[triallist_no][trial][3];
  data.stim_base_tilt = triallist[triallist_no][trial][4];  // tilt left or right array
  data.tilt_more_to_left = triallist[triallist_no][trial][5];
  data.stim_tilt = computeTrialTilt(triallist[triallist_no][trial][3], triallist[triallist_no][trial][4], triallist[triallist_no][trial][5]);
};

function createPerceptRespScreen(canvas){
  var window_dims = getScreenDims();
  var triallist_no = getCurrentTriallist();
  var trial = getCurrentTrialNo();
  var ctx = canvas.getContext('2d');
  var stimSize = triallist[triallist_no][trial][0]

  // fixation cross
  ctx.strokeStyle = 'white';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(window_dims[1]/2-10, window_dims[0]/2);
  ctx.lineTo(window_dims[1]/2+10, window_dims[0]/2);
  ctx.moveTo(window_dims[1]/2, window_dims[0]/2-10);
  ctx.lineTo(window_dims[1]/2, window_dims[0]/2+10);
  ctx.stroke();

  // draw surrouding circle
  ctx.beginPath();
  ctx.strokeStyle = 'white';
  ctx.lineWidth = 2;
  ctx.arc(window_dims[1]/2, window_dims[0]/2, stimSize/2, 0, 2*Math.PI);
  ctx.stroke();

  // draw reference line at +/-45 degree, always draw middle level size  circle
  var ref_coor_start = getRefLineCoordinates(triallist[triallist_no][trial][4], stimSize, window_dims, true);
  var ref_coor_end = getRefLineCoordinates(triallist[triallist_no][trial][4], stimSize, window_dims, false);
  ctx.beginPath()
  ctx.strokeStyle = '#ff0000';
  ctx.lineWidth = 2;
  ctx.moveTo(ref_coor_start[0], ref_coor_start[1]);
  ctx.lineTo(ref_coor_end[0], ref_coor_end[1]);
  ctx.stroke();

  ctx.textAlign = "center";
  ctx.font = "24px Arial";
  ctx.fillStyle = 'white';
  ctx.fillText("←      OR       →", window_dims[1]/2, window_dims[0]/2 + window_dims[0]/2/2);
};

function createConfRespScreen(canvas){
  var window_dims = getScreenDims();
  var ctx = canvas.getContext('2d');

  // fixation cross
  ctx.strokeStyle = 'white';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(window_dims[1]/2-10, window_dims[0]/2);
  ctx.lineTo(window_dims[1]/2+10, window_dims[0]/2);
  ctx.moveTo(window_dims[1]/2, window_dims[0]/2-10);
  ctx.lineTo(window_dims[1]/2, window_dims[0]/2+10);
  ctx.stroke();

  ctx.textAlign = "center";
  ctx.font = "24px Arial";
  ctx.fillStyle = 'white';
  ctx.fillText("CONFIDENCE", window_dims[1]/2, window_dims[0]/2 + window_dims[0]/2/4);
  ctx.fillText("low               high", window_dims[1]/2, window_dims[0]/2 + window_dims[0]/2/2.5);
  ctx.fillText(" 1     2     3    4", window_dims[1]/2, window_dims[0]/2 + window_dims[0]/2/2);
};

function createFeedbackScreen(){
  var is_it_correct = jsPsych.data.get().last(2).values()[0].correct;
  var feedback;
  if (is_it_correct === 1){
    feedback = '<p style="font-size:40px;color:#00ff00;"> CORRECT </p>'
  } else {
    feedback = '<p style="font-size:40px;color:#ff0000;"> WRONG </p>'
  };
  return feedback;
};

// These are created for staircase procedure with SC ending
function createGaborSC(canvas){
  var size = 150;
  var contrast = 0.75;
  var tilt_diff = jsPsych.data.get().last(2).values()[0].next_tilt;
  if (typeof tilt_diff === 'undefined'){
    tilt_diff = initial_tilt;
  };
  var tilt_left = 45;
  var tilt_more = jsPsych.randomization.sampleWithoutReplacement([true, false], 1)[0];
  jsPsych.data.addProperties({tilt_diff: tilt_diff, tilt_left: tilt_left, tilt_more: tilt_more});

  var ctx = canvas.getContext('2d');
  var window_dims = getScreenDims();
  // var triallist_no = getCurrentTriallist();
  // var trial = getCurrentTrialNo();
  var stimSize = size;
  var contrast = contrast;
  var SD_mask = stimSize/8;
  var numCyclesPer_stimSize = 8;   // for gabor stimulus, number of cycels per stimulus size
  var trial_tilt = computeTrialTilt(tilt_diff, tilt_left, tilt_more)

  var arr = jglMakeGrating(stimSize, stimSize, numCyclesPer_stimSize, trial_tilt, 0);
  var mask = make2dMask(arr, 1, SD_mask);
  var gabor = jglCreateGabor(ctx, arr, mask, contrast);
  var offset = (window_dims[0] - window_dims[1])/2
  ctx.putImageData(gabor, (window_dims[0] - stimSize)/2 - offset , (window_dims[1] - stimSize)/2 + offset); //display in the center
};

function createPerceptRespScreenSC(canvas){
  var window_dims = getScreenDims();
  var triallist_no = getCurrentTriallist();
  var trial = getCurrentTrialNo();
  var ctx = canvas.getContext('2d');
  var tilt_diff = jsPsych.data.getLastTrialData().select('tilt_diff').values[0];
  var tilt_left = jsPsych.data.getLastTrialData().select('tilt_left').values[0];
  var tilt_more = jsPsych.data.getLastTrialData().select('tilt_more').values[0];
  var stimSize = 150

  // fixation cross
  ctx.strokeStyle = 'white';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(window_dims[1]/2-10, window_dims[0]/2);
  ctx.lineTo(window_dims[1]/2+10, window_dims[0]/2);
  ctx.moveTo(window_dims[1]/2, window_dims[0]/2-10);
  ctx.lineTo(window_dims[1]/2, window_dims[0]/2+10);
  ctx.stroke();

  // draw surrouding circle
  ctx.beginPath();
  ctx.strokeStyle = 'white';
  ctx.lineWidth = 2;
  ctx.arc(window_dims[1]/2, window_dims[0]/2, stimSize/2, 0, 2*Math.PI);
  ctx.stroke();

  // draw reference line at +/-45 degree, always draw middle level size  circle
  var ref_coor_start = getRefLineCoordinates(tilt_left, stimSize, window_dims, true);
  var ref_coor_end = getRefLineCoordinates(tilt_left, stimSize, window_dims, false);
  ctx.beginPath()
  ctx.strokeStyle = '#ff0000';
  ctx.lineWidth = 2;
  ctx.moveTo(ref_coor_start[0], ref_coor_start[1]);
  ctx.lineTo(ref_coor_end[0], ref_coor_end[1]);
  ctx.stroke();

  ctx.textAlign = "center";
  ctx.font = "24px Arial";
  ctx.fillStyle = 'white';
  ctx.fillText("←      OR       →", window_dims[1]/2, window_dims[0]/2 + window_dims[0]/2/2);
};

function createStartPracticeScreen(){
  var instruction;
  var trial_no = jsPsych.data.getLastTrialData().select('trial_no').values[0];
  var curr_triallist = jsPsych.data.getLastTrialData().select('curr_triallist').values[0];
  if (typeof curr_triallist === "undefined"){   // this will catch for the first trial
    curr_triallist = 0;
  } else if (triallist_length[curr_triallist] === trial_no+1){
    curr_triallist += 1;
  };

  switch (curr_triallist){
    case 0:
      instruction = start_prac_instruct_1;
      break;
    case 1:
      instruction = start_prac_instruct_2;
      break;
    case 2:
      instruction = start_prac_instruct_3;
      break;
  };
 return instruction;
};

function createStartBreakScreen(){
  var current_trial_no = getCurrentTrialNo() + 1;
  var current_triallist = getCurrentTriallist();
  var total_no_of_trials = triallist[current_triallist].length;

  var break_text =
    '<p style="font-size:20px;color:#000000">' +
    'You have completed ' + Math.round(current_trial_no/total_no_of_trials*100) +
    '\% of the experiment.' +
    // '<br><br> RUN: ' + run_no + '<br><br> BLOCK: ' + block_no +
    '<br><br>Take a short break.' +
    '<br><br>When you are ready, press spacebar to proceed.'

  return break_text;
};

function createEndBreakScreen(){
  var current_trial_no = getCurrentTrialNo() + 1;
  var current_triallist = getCurrentTriallist();
  var total_no_of_trials = triallist[current_triallist].length;
  var run_no = getCurrentRunBlockNo(current_trial_no, total_no_of_trials)[0];
  var block_no = getCurrentRunBlockNo(current_trial_no, total_no_of_trials)[1];

  var run_block_no_text =
    '<p style="font-size:32px;color:#000000">' +
    '<br><br> RUN: &nbsp ' + run_no + '/5 <br><br> BLOCK: ' + block_no + '/6 ';

  return run_block_no_text;
};
