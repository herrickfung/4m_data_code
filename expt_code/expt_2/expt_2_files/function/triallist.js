// parameters for staircase
var initial_tilt = 2;
var step = 0.2;
var up_down_factor = 0.7393;   // from Garcı́a-Pérez, M. A. (1998). Forced-choice staircases with fixed step sizes: Asymptotic and small-sample properties. Vision Research, 38(12), 1861–1881. https://doi.org/10.1016/S0042-6989(97)00340-4
var max_reversal = 14;
var max_trial_SC = 60;
var optimal1;
var optimal2;

function createTriallist(is_test, optimal){
  // trial_array [size, duration, contrast, spatial_freq, tilt_diff, tilt_left, tilt_more]
  // declare stimulus variables for creating triallist
  var triallist = []
  var size_array = [90, 225];    // equivalent to [3, 7.5] in visual angle
  var dur = 200  // stimulus duration
  var contrast_array = [0.85, 0.75];
  var spatial_freq_array = [3.5, 8];
  var tilt_diff_array = [optimal, optimal*1.8];  // change later to [optimal/2, optimal, optimal * 2]
  var tilt_left = 45;  // - is left; + is right
  var tilt_more_to_the_left_array = [true, false];  // if true, tilt more to the left
  if (is_test === false){
    var no_of_observations = 25;  // together with tilt_more_to_the_left, it is 50 per observation.
  } else {
    var no_of_observations = 1;
  };

  // create triallist
  for (let reps = 0; reps < no_of_observations; reps++){
    for (var size of size_array){
      for (var contrast of contrast_array){
        for (var sf of spatial_freq_array){
          for (var tilt_diff of tilt_diff_array){
            for (var tilt_more of tilt_more_to_the_left_array){
              trial = [size, dur, contrast, sf, tilt_diff, tilt_left, tilt_more];
              triallist.push(trial);
            };
          };
        };
      };
    };
  };
  return triallist;
};

function breakTraillist(is_test, total_no_of_trials, trial_to_break){
  var no_of_breaks = Math.round(total_no_of_trials / trial_to_break);
  var break_trial_array = [];
  for (let i = 1; i < no_of_breaks; i++){
    break_trial_no = (total_no_of_trials / no_of_breaks * i);
    break_trial_no = Math.round(break_trial_no);
    break_trial_array.push(break_trial_no);
  };
  return break_trial_array;
};

function createPracticeTriallist1(){
  // first practice
  var triallist = [];
  var size = 150;
  var dur = 200;
  var contrast = 0.75;
  var sf = 8;
  var tilt_diff = 10;
  var tilt_left = 45;
  var tilt_more_array = [true, false];  // if true, tilt more to the left
  var no_of_observations = 15;
  // trial_array [size, duration_array, contrast, sf, tilt_diff, tilt_left, tilt_more]
  for (let i = 0; i < no_of_observations; i++){
    var ran1 = getRandomInt(0,2);
    var trial = [size, dur, contrast, sf, tilt_diff, tilt_left, tilt_more_array[ran1]];
    triallist.push(trial);
  };
  return triallist;
};

function createPracticeTriallist2(){
  // second practice
  var triallist = [];
  var size = 150;
  var dur = 200;
  var contrast = 0.75;
  var sf = 8;
  var tilt_diff_array = [5,3,2,1];
  var tilt_left = 45;
  var tilt_more_array = [true, false];  // if true, tilt more to the left
  var no_of_observations = 5;
  // trial_array [size, duration_array, contrast, sf, tilt_diff, tilt_left, tilt_more]
  for (let j = 0; j < tilt_diff_array.length; j++){
    for (let i = 0; i < no_of_observations; i++){
      var ran1 = getRandomInt(0,2);
      var trial = [size, dur, contrast, sf, tilt_diff_array[j], tilt_left, tilt_more_array[ran1]];
      triallist.push(trial);
    };
  };
  return triallist;
};

function createPracticeTriallist3(){
  // second practice
  var triallist = [];
  var size = 150;
  var dur = 200;
  var contrast = 0.75;
  var sf = 8;
  var tilt_diff_array = [1,2,4];
  var tilt_left = 45;
  var tilt_more_array = [true, false];  // if true, tilt more to the left
  var no_of_observations = 20;
  // trial_array [size, duration_array, contrast, sf, tilt_diff, tilt_left, tilt_more]
  for (let i = 0; i < no_of_observations; i++){
    var ran1 = getRandomInt(0,2);
    var ran2 = getRandomInt(0,3);
    var trial = [size, dur, contrast, sf, tilt_diff_array[ran2], tilt_left, tilt_more_array[ran1]];
    triallist.push(trial);
  };
  return triallist;
};

function createPracticeTriallist4(optimal_tilt){
  // second day practice
  var triallist = [];
  var size = 150;
  var dur = 200;
  var contrast = 0.75;
  var sf = 8;
  var tilt_diff_array = [optimal_tilt, optimal_tilt*1.8];
  var tilt_left = 45;
  var tilt_more_array = [true, false];  // if true, tilt more to the left
  var no_of_observations = 24;
  // trial_array [size, duration_array, contrast, sf, tilt_diff, tilt_left, tilt_more]
  for (let i = 0; i < no_of_observations; i++){
    var ran1 = getRandomInt(0,2);
    var ran2 = getRandomInt(0,2);
    var trial = [size, dur, contrast, sf, tilt_diff_array[ran2], tilt_left, tilt_more_array[ran1]];
    triallist.push(trial);
  };
  return triallist;
};

