// parameters for staircase
var initial_tilt = 2;
var step = 0.2;
var up_down_factor = 0.7393;   // from Garcı́a-Pérez, M. A. (1998). Forced-choice staircases with fixed step sizes: Asymptotic and small-sample properties. Vision Research, 38(12), 1861–1881. https://doi.org/10.1016/S0042-6989(97)00340-4
var max_reversal = 14;
var max_trial_SC = 60;
var optimal1;
var optimal2;

function createTriallist(is_test, optimal){
  // trial_array [size, duration_array, contrast, tilt_diff, tilt_left, tilt_more]
  // declare stimulus variables for creating triallist
  var triallist = []
  var size_array = [75, 150, 225];    // equivalent to [2.5, 5, 7.5] in visual angle
  var duration_array = [33, 100, 500];
  var contrast_array = [0.9, 0.75, 0.1];
  var tilt_diff_array = [optimal/2, optimal, optimal*2];  // change later to [optimal/2, optimal, optimal * 2]
  var tilt_left_or_right_array = [45];  // - is left; + is right
  var tilt_more_to_the_left_array = [true, false];  // if true, tilt more to the left
  var no_of_observations = 36;  // together with tilt_more_to_the_left, it is 72 per observation.

  if (is_test === false){
    for (let i = 0; i < no_of_observations; i++){
      for (let j = 0; j < tilt_left_or_right_array.length; j++){
        for (let k = 0; k < tilt_more_to_the_left_array.length; k++){
          // baseline conditions, done twice.
          var trial = [size_array[1], duration_array[1], contrast_array[1], tilt_diff_array[1], tilt_left_or_right_array[j], tilt_more_to_the_left_array[k]];
          triallist.push(trial);
          triallist.push(trial);
          // size manipulation trial
          var trial = [size_array[0], duration_array[1], contrast_array[1], tilt_diff_array[1], tilt_left_or_right_array[j], tilt_more_to_the_left_array[k]];
          triallist.push(trial);
          var trial = [size_array[2], duration_array[1], contrast_array[1], tilt_diff_array[1], tilt_left_or_right_array[j], tilt_more_to_the_left_array[k]];
          triallist.push(trial);
          // duration_array manipulation trial
          var trial = [size_array[1], duration_array[0], contrast_array[1], tilt_diff_array[1], tilt_left_or_right_array[j], tilt_more_to_the_left_array[k]];
          triallist.push(trial);
          var trial = [size_array[1], duration_array[2], contrast_array[1], tilt_diff_array[1], tilt_left_or_right_array[j], tilt_more_to_the_left_array[k]];
          triallist.push(trial);
          // contrast manipulation trial
          var trial = [size_array[1], duration_array[1], contrast_array[0], tilt_diff_array[1], tilt_left_or_right_array[j], tilt_more_to_the_left_array[k]];
          triallist.push(trial);
          var trial = [size_array[1], duration_array[1], contrast_array[2], tilt_diff_array[1], tilt_left_or_right_array[j], tilt_more_to_the_left_array[k]];
          triallist.push(trial);
          // tilt difference manipulation trial
          var trial = [size_array[1], duration_array[1], contrast_array[1], tilt_diff_array[0], tilt_left_or_right_array[j], tilt_more_to_the_left_array[k]];
          triallist.push(trial);
          var trial = [size_array[1], duration_array[1], contrast_array[1], tilt_diff_array[2], tilt_left_or_right_array[j], tilt_more_to_the_left_array[k]];
          triallist.push(trial);
        };
      };
    };
  }

  else {
    // run through all conditions, for checking
    var triallist = [
        [size_array[0], duration_array[1], contrast_array[1], tilt_diff_array[1], 45, true],
        [size_array[1], duration_array[1], contrast_array[1], tilt_diff_array[1], 45, false],
        [size_array[2], duration_array[1], contrast_array[1], tilt_diff_array[1], 45, true],
        [size_array[1], duration_array[0], contrast_array[1], tilt_diff_array[1], 45, false],
        [size_array[1], duration_array[1], contrast_array[1], tilt_diff_array[1], 45, true],
        [size_array[1], duration_array[2], contrast_array[1], tilt_diff_array[1], 45, false],
        [size_array[1], duration_array[1], contrast_array[0], tilt_diff_array[1], 45, true],
        [size_array[1], duration_array[1], contrast_array[1], tilt_diff_array[1], 45, false],
        [size_array[1], duration_array[1], contrast_array[2], tilt_diff_array[1], 45, true],
        [size_array[1], duration_array[1], contrast_array[0], tilt_diff_array[0], 45, false],
        [size_array[1], duration_array[1], contrast_array[1], tilt_diff_array[1], 45, true],
        [size_array[1], duration_array[1], contrast_array[2], tilt_diff_array[2], 45, false],
    ];
    triallist =triallist.flatMap(i => [i,i]);
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
  var size_array = [150];
  var duration_array = [100];
  var contrast_array = [0.75];
  var tilt_diff_array = [10];  // change later to [optimal/2, optimal, optimal * 2]
  var tilt_left_or_right_array = [45];  // - is left; + is right
  var tilt_more_to_the_left_array = [true, false];  // if true, tilt more to the left
  var no_of_observations = 15;
  // trial_array [size, duration_array, contrast, tilt_diff, tilt_left, tilt_more]
  for (let i = 0; i < no_of_observations; i++){
    var ran1 = getRandomInt(0,2);
    var trial = [size_array[0], duration_array[0], contrast_array[0], tilt_diff_array[0], tilt_left_or_right_array[0], tilt_more_to_the_left_array[ran1]];
    triallist.push(trial);
  };
  return triallist;
};

function createPracticeTriallist2(){
  // second practice
  var triallist = [];
  var size_array = [150];
  var duration_array = [100];
  var contrast_array = [0.75];
  var tilt_diff_array = [5,3,2,1];  // change later to [optimal/2, optimal, optimal * 2]
  var tilt_left_or_right_array = [45];  // - is left; + is right
  var tilt_more_to_the_left_array = [true, false];  // if true, tilt more to the left
  var no_of_observations = 5;
  // trial_array [size, duration_array, contrast, tilt_diff, tilt_left, tilt_more]
  for (let j = 0; j < tilt_diff_array.length; j++){
    for (let i = 0; i < no_of_observations; i++){
      var ran1 = getRandomInt(0,2);
      var trial = [size_array[0], duration_array[0], contrast_array[0], tilt_diff_array[j], tilt_left_or_right_array[0], tilt_more_to_the_left_array[ran1]];
      triallist.push(trial);
    };
  };
  return triallist;
};

function createPracticeTriallist3(){
  // third practice
  var triallist = [];
  var size_array = [150];
  var duration_array = [100];
  var contrast_array = [0.75];
  var tilt_diff_array = [1,2,4];  // change later to [optimal/2, optimal, optimal * 2]
  var tilt_left_or_right_array = [45];  // - is left; + is right
  var tilt_more_to_the_left_array = [true, false];  // if true, tilt more to the left
  var no_of_observations = 20;
  // trial_array [size, duration_array, contrast, tilt_diff, tilt_left, tilt_more]
  for (let i = 0; i < no_of_observations; i++){
    var ran2 = getRandomInt(0,2);
    var ran3 = getRandomInt(0,3);
    var trial = [size_array[0], duration_array[0], contrast_array[0], tilt_diff_array[ran3], tilt_left_or_right_array[0], tilt_more_to_the_left_array[ran2]];
    triallist.push(trial);
  };
  return triallist;
};

