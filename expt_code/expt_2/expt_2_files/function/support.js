
// these are support function

function shuffle(array) {
  // this will shuffle all triallist
  // Fisher-Yates algorithm found on web
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  };
  return array;
};

function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min)) + min;
};

function getDatetime(){
  // get date and time for record in data
  const datetime = new Date().toString();
  return datetime;
};

function computeTrialTilt(tilt_diff, base_tilt, more_left){
  // this function compute the trial tilt, mainly for data storage
  if (more_left === true){
    var computed_trial_tilt = base_tilt + tilt_diff;
  } else {
    var computed_trial_tilt = base_tilt - tilt_diff;
  };
  return computed_trial_tilt;
};

function getRefLineCoordinates(trial_ref, stimSize, window, start_coor){
  // this function gets the coordinates for the red 45 degree reference line
  // use whenever a ref line is needed
  var line_length = 20
  if (start_coor === true){
    x_coor = window[1]/2 + (stimSize-line_length)/2 * Math.cos(Math.PI/4) * trial_ref/45;
    y_coor = window[0]/2 - (stimSize-line_length)/2 * Math.sin(Math.PI/4);
  } else {
    x_coor = window[1]/2 + (stimSize+line_length)/2 * Math.cos(Math.PI/4) * trial_ref/45;
    y_coor = window[0]/2 - (stimSize+line_length)/2 * Math.sin(Math.PI/4);
  };
  return [x_coor, y_coor];
};

function getCurrentTrialNo(){
  // get the trial number from last trial, for reading data from triallist
  // use in main experiment loop except for fixation
  var curr_trial_no = jsPsych.data.getLastTrialData().select('trial_no').values[0];
  return curr_trial_no;
};

function updateTrialNo(){
  // this is only used in the fixation function
  // align the trial no. with the main experiment loop
  var trial_no = jsPsych.data.getLastTrialData().select('trial_no').values[0];
  if (typeof trial_no === "undefined"){   // this will catch for the first trial
    trial_no = 0;
  } else {    // the rest will add one and align with the loop
    trial_no += 1;
  };
  return trial_no;
};

function getCurrentTriallist(){
  // get the triallist from last trial, for reading data from triallist
  // use in main experiment loop except for fixation
  var curr_triallist = jsPsych.data.getLastTrialData().select('curr_triallist').values[0];
  return curr_triallist;
};

function updateTriallist(){
  // this is only used in the fixation function
  // update the triallist number
  var trial_no = jsPsych.data.getLastTrialData().select('trial_no').values[0];
  var curr_triallist = jsPsych.data.getLastTrialData().select('curr_triallist').values[0];
  if (typeof curr_triallist === "undefined"){   // this will catch for the first trial
    curr_triallist = 0;
  } else if (triallist_length[curr_triallist] === trial_no+1){
    curr_triallist += 1;
  } else {
    curr_triallist = getCurrentTriallist();
  };
  return curr_triallist;
};

function getCurrentRunBlockNo(trial_no, total_no_of_trials){
  // this compute run and block no to show in break screen
  // 5 runs 6 blocks
  var no_of_runs = 5;
  var no_of_blks = 5;
  var trial_in_run = no_of_blks * trials_to_break;
  var trial_in_blk = trials_to_break;

  var current_run = Math.floor((trial_no / trial_in_run) + 1);
  var current_blk = (trial_no - (Math.floor(trial_no / trial_in_run)) * trial_in_run)/trial_in_blk + 1;
  return [current_run, current_blk];
};

function checkAccuracy(data){
  // check if the response is correct
  // used in percept_screen for data storage
  var triallist_no = getCurrentTriallist();
  var trial = getCurrentTrialNo();
  // this will align data response with triallist[5] answer,
  // more left is true, more right is false
  if (data.response === 'arrowleft'){
    data.resp_check = true;
  } else {
    data.resp_check = false;
  };
  if (data.resp_check == triallist[triallist_no][trial][6]){
    data.correct = 1;
  } else {
    data.correct = 0;
  };
};

function getTrialParameter(){
  // use in perceptual response sc
  var tilt_diff = jsPsych.data.get().last(2).values()[0].tilt_diff;
  var tilt_more = jsPsych.data.get().last(2).values()[0].tilt_more;
  return [tilt_diff, tilt_more];
};

function updateSCParameter(trial_tilt, trial_accuracy){
  // get previous staircase parameter
  var num_of_correct = jsPsych.data.get().last(4).values()[0].num_correct;
  var reversal_count = jsPsych.data.get().last(4).values()[0].reversal_count;
  var make_easier = jsPsych.data.get().last(4).values()[0].make_easier;
  var tilt_record = jsPsych.data.get().last(4).values()[0].tilt_record;
  var trial_count = jsPsych.data.get().last(4).values()[0].trial_no;
  var next_tilt;

  if (typeof num_of_correct === 'undefined'){
    // initial parameter for staircase
    // use in the first staircase trial
    num_of_correct = 0;
    reversal_count = 0;
    make_easier = true;
    tilt_record = [];
    trial_count = 0;
  }

  if (trial_accuracy === 1){     // correct
    if (num_of_correct < 2){
      num_of_correct += 1;
      next_tilt = trial_tilt;
    } else {
      if (make_easier === true){
        make_easier = false;
        reversal_count += 1;
        tilt_record.push(trial_tilt);
      }
      next_tilt = trial_tilt - step;
      if (next_tilt < 0){
        next_tilt = 0
      }
      num_of_correct = 0
    };
  } else {                      // wrong
    if (make_easier === false){
      make_easier = true;
      reversal_count += 1;
      tilt_record.push(trial_tilt);
    };
    next_tilt = trial_tilt + step / up_down_factor;
    num_of_correct = 0;
  };
  trial_count += 1;
  return [num_of_correct, reversal_count, make_easier, next_tilt, tilt_record, trial_count];
};

function checkAccuracySC(data){
  // check if the response is correct
  // update staircase
  var [tilt_diff, tilt_more] = getTrialParameter();
  // this will align data response with triallist[5] answer,
  // more left is true, more right is false
  if (data.response === 'arrowleft'){
    data.resp_check = true;
  } else {
    data.resp_check = false;
  };
  if (data.resp_check === tilt_more){
    data.correct = 1;
  } else {
    data.correct = 0;
  };

  var staircase_parameter = updateSCParameter(tilt_diff, data.correct);
  data.num_correct = staircase_parameter[0];
  data.reversal_count = staircase_parameter[1];
  data.make_easier = staircase_parameter[2];
  data.next_tilt = staircase_parameter[3];
  data.tilt_record = staircase_parameter[4];
  data.trial_no =  staircase_parameter[5];
};

// add min_reversal to be 8, so we can average at least 7 numbers
function endSC(data){
  var reversal_count = data.values()[2].reversal_count;
  var trial_count = data.values()[2].trial_no;
  console.log(reversal_count)
  if (!(reversal_count < max_reversal) || !(trial_count < max_trial_SC)) {
    return false;
  } else {
    return true;
  };
};

function updateOptimal1(){
  tilt_record = jsPsych.data.get().last(1).values()[0].tilt_record;
  tilt_record = tilt_record.filter((_, i) => i >= 1)
  var average = tilt_record.reduce((a, b) => a + b, 0) / tilt_record.length;
  var final = [];
  final.push(average);
  optimal1 = final[final.length-1];
};

function updateOptimal2(){
  tilt_record = jsPsych.data.get().last(1).values()[0].tilt_record;
  tilt_record = tilt_record.filter((_, i) => i >= 1)
  var average = tilt_record.reduce((a, b) => a + b, 0) / tilt_record.length;
  var final = [];
  final.push(average);
  optimal2 = final[final.length-1];
};

function updateExperimentalTriallist(){
  var optimal_tilt;
  var subject_id = getSubjNo()
  if (typeof optimal1 === 'undefined' || isNaN(optimal1)){
    optimal_tilt = 2;
  } else {
  optimal_tilt = (optimal1+optimal2)/2;
  };
  triallist[3] = createTriallist(is_test, optimal_tilt),
  shuffle(triallist[3]);
  saveData("stim_var_expt_2_data/" + subject_id + "_optimal.txt", optimal_tilt)
};

function updateTriallistD2(){
  var optimal_tilt = getOptimal();
  triallist[1] = createPracticeTriallist4(optimal_tilt);
  triallist[2] = createTriallist(is_test, optimal_tilt);
  console.log(triallist[1]);
  console.log(triallist[2]);
  shuffle(triallist[1]);
  shuffle(triallist[2]);
};

function getOptimal(){
  var optimal_tilt = new XMLHttpRequest();
  var subject_id = getSubjNo()
  optimal_tilt.open("GET", "data/stim_var_expt_2_data/" + subject_id + "_optimal.txt", false);
  optimal_tilt.send();
  optimal_tilt = Number(optimal_tilt.response);
  return optimal_tilt;
};

function getSubjNo(){
  var subject_id = jsPsych.data.get().filter({trial_type: "survey"});
  subject_id = Object.values(subject_id.select('response').values[0])[0];
  return subject_id
};

function saveData(filename, filedata){
  $.ajax({
    type: 'post',
    cache: false,
    url: 'savedata.php',
    data: {filename: filename, filedata: filedata}
  });
};
