// INFO
// stimulus variability project - experiment code
// triallist.js — create triallist, stimulus parameter, setup info.
// stimulus.js — contains all stimulus function
// support.js — contains all other support function
// gaborFunctions.js — for creating Gabor stimuli
// instruction.js — contains all long text elements

// these defines all triallist
// triallist[3] will be updated before experimental trials at start_experiment: on_start function

const is_test = false;   // true for test, false for real
var trials_to_break = 32;  //define how many trials to have break in expeirmental trials
var triallist_length = [];
var triallist = [createPracticeTriallist1(),
                 createPracticeTriallist4(),
                 createTriallist(is_test),
                ];
shuffle(triallist[2]);
var break_trial_array = breakTraillist(is_test, triallist[2].length, trials_to_break);
for (i=0; i < triallist.length; i++){
  list_length = triallist[i].length;
  triallist_length.push(list_length);
};

const jsPsych = initJsPsych({
  override_safe_mode: true,
  on_finish: function(){
    jsPsych.data.get().localSave('csv', "SVE2_" + subject_id + "_d2.csv")
    window.location = "https://app.prolific.com/submissions/complete?cc=CHZ22XMB";
  },
});

function getScreenDims(){
  var chinrest_data = jsPsych.data.get().filter({trial_type: "virtual-chinrest"});
  var pixel2degree = parseFloat(chinrest_data.select('px2deg').values);
  var window_width = parseFloat(chinrest_data.select('win_width_deg').values);
  var window_height = parseFloat(chinrest_data.select('win_height_deg').values);
  window_width = window_width * pixel2degree / 2;
  window_height = window_height * pixel2degree / 2;
  return [window_width, window_height];
};

////////////////////////////////////////////////////////////////////////////////
// main stimulus define for timeline

// main fixation stimulus_screen
var fixation = {
  type: jsPsychCanvasKeyboardResponse,
  stimulus: createFixation,
  choices: 'none',
  trial_duration: 1000,
  is_html: true,
  canvas_size: getScreenDims,
  data: {
    datetime: getDatetime,
    subjectID: getSubjNo,
    curr_triallist: getCurrentTriallist,
    trial_no: updateTrialNo,
    stimulus: "fixation",
  },
};

// this creates fixation without updating trial_no
// use before or after main fixation are both fine
var fixation_no_update = {
  type: jsPsychCanvasKeyboardResponse,
  stimulus: createFixation,
  choices: 'none',
  trial_duration: 1000,
  is_html: true,
  canvas_size: getScreenDims,
  data: {
    datetime: getDatetime,
    subjectID: getSubjNo,
    curr_triallist: getCurrentTriallist,
    trial_no: getCurrentTrialNo,  // here's the only difference
    stimulus: "fixation",
  },
};

// gabor stimulus screen
var gabor = {
  type: jsPsychCanvasKeyboardResponse,
  stimulus: createGabor,
  choices: 'NO_KEYS',
  is_html: true,
  stimulus_duration: 0, //  on_start update
  trial_duration: 0,    //  on_start update
  response_ends_trial: true,
  canvas_size: getScreenDims,
  on_start: createGaborStart,
  on_finish: createGaborEnd,
  data: {
    datetime: getDatetime,
    subjectID: getSubjNo,
    curr_triallist: getCurrentTriallist,
    trial_no: getCurrentTrialNo,
    stimulus: "gabor-stimuli",
    // on_finish_update for all below
    stim_size: 0,
    stim_dur: 0,
    stim_contrast: 0,
    stim_sf: 0,
    stim_tilt_diff: 0,
    stim_base_tilt: 0,
    stim_tilt_more_to_left: 0,
    stim_tilt: 0,
  },
};

// perceptual responses screen
var percept_resp = {
  type: jsPsychCanvasKeyboardResponse,
  stimulus: createPerceptRespScreen,
  choices: ['ArrowLeft', 'ArrowRight'],
  is_html: true,
  canvas_size: getScreenDims,
  on_finish: checkAccuracy,
  data: {
    datetime: getDatetime,
    subjectID: getSubjNo,
    curr_triallist: getCurrentTriallist,
    trial_no: getCurrentTrialNo,
    stimulus: "perceptual_response",
    // on_finish_update for all below
    resp_check: 0,
    correct: 0,
  },
};

// confidence resopnses screen
var conf_resp = {
  type: jsPsychCanvasKeyboardResponse,
  stimulus: createConfRespScreen,
  choices: ['1','2','3','4'],
  is_html: true,
  canvas_size: getScreenDims,
  data: {
    datetime: getDatetime,
    subjectID: getSubjNo,
    curr_triallist: getCurrentTriallist,
    trial_no: getCurrentTrialNo,
    stimulus: "confidence_response",
  },
};

var feedback = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: createFeedbackScreen,
  choices: 'NO_KEYS',
  trial_duration: 500,
  data: {
    datetime: getDatetime,
    subjectID: getSubjNo,
    curr_triallist: getCurrentTriallist,
    trial_no: getCurrentTrialNo,
    stimulus: "feedback",
  },
};

var start_experiment = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: start_experiment_instruct_d2,
  choices: " ",
  data: {
    datetime: getDatetime,
    subjectID: getSubjNo,
    curr_triallist: 2,
    trial_no: -1,
    stimulus: "start experiment"
  },
};

////////////////////////////////////////////////////////////////////////////////
// other minor variables in timeline

const preload = {
  type: jsPsychPreload,
  images: ['expt_2_files/img/card.png',
           'expt_2_files/img/gabor_45.png',
           'expt_2_files/img/gabor_left.png',
           'expt_2_files/img/gabor_right.png',
           'expt_2_files/img/resp_screen.png',
           ]
};

const browser_check = {
  type: jsPsychBrowserCheck,
  inclusion_function: (data) => {
    return ['chrome', 'firefox'].includes(data.browser) && data.mobile === false;
  },
  exclusion_message: (data) => {
    if (data.mobile === true){
      return '<p> You must use a desktop/laptop computer to participate in this experiment. </p>'
    } else if (!(['chrome', 'firefox'].includes(data.browser))){
      return '<p> You must use Chrome or Firefox as your browser to complete this experiment. </p>'
    }
  }
};

const getPartiInfo = {
  type: jsPsychSurvey,
  pages: parti_info_questions_d2,
  title: "Please provide the following information.",
  on_finish: updateTriallistD2,
};

const confirmPartiInfo = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: createConfirmInfoScreen,
  choices: "+",
  data: {
    datetime: getDatetime,
    subjectID: getSubjNo,
    stimulus: "confirm parti info"
  },
};

const welcome = {
  type: jsPsychInstructions,
  pages: welcome_text_d2,
  show_clickable_nav: true,
  show_page_number: true,
};

const chinrest = {
  type: jsPsychVirtualChinrest,
  blindspot_reps: 3,
  resize_units: "deg",
  pixels_per_unit: 30,
  item_path: "expt_2_files/img/card.png",
  viewing_distance_report: 'none',
};

var start_practice_instruction = {
  type: jsPsychInstructions,
  pages: task_instruction,
  show_clickable_nav: true,
  show_page_number: true,
};

var start_practice = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: createStartPracticeScreenD2,
  choices: " ",
  data: {
    datetime: getDatetime,
    subjectID: getSubjNo,
    trial_no: -1,
    curr_triallist: updateTriallist,
    stimulus: "start_practice"
  },
};

var start_break_procedure = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: createStartBreakScreen,
  choices: " ",
  data: {
    datetime: getDatetime,
    subjectID: getSubjNo,
    curr_triallist: getCurrentTriallist,
    trial_no: getCurrentTrialNo,
    stimulus: "start break"
  },
};

var end_break_procedure = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: createEndBreakScreen,
  choices: " ",
  trial_duration: 1000,
  data: {
    datetime: getDatetime,
    subjectID: getSubjNo,
    curr_triallist: getCurrentTriallist,
    trial_no: getCurrentTrialNo,
    stimulus: "start break"
  },
};

var debriefing = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: debrief_text_d2,
  choices: "NO_KEYS",
  trial_duration: 60000,
  on_start: function(){
    var subject_id = getSubjNo();
    saveData("data/SVE2_" + subject_id + "_d2.csv", jsPsych.data.get().csv());
  },
};

var debriefing2 = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: debrief_text_d2,
  choices: "NO_KEYS",
  trial_duration: 5000,
};

var startFullScreen = {
  type: jsPsychFullscreen,
  fullscreen_mode: true,
};

var endFullScreen = {
  type: jsPsychFullscreen,
  fullscreen_mode: false,
};

// end of define variables
////////////////////////////////////////////////////////////////////////////////
// the main experiment timeline starts here
var timeline = [];

timeline.push(browser_check);
timeline.push(getPartiInfo);
timeline.push(confirmPartiInfo);
timeline.push(welcome);
timeline.push(startFullScreen);
timeline.push(chinrest);

timeline.push(start_practice_instruction);
//////////////////////////////////////////////////////////////////////////////
 // run 2 practice triallist
 for (tl_no = 0; tl_no < 2; tl_no++){   // tl_no is just triallist_no
  timeline.push(start_practice);
  for (let trial = 0; trial < triallist_length[tl_no]; trial++) {
    // extend fixation only for the first trial
    if (trial === 0){
      timeline.push(fixation_no_update);
    };
    timeline.push(fixation);
    timeline.push(gabor);
    timeline.push(percept_resp);
    timeline.push(conf_resp);
    // give feedback for the first triallist
    if (tl_no != 1){
      timeline.push(feedback);
    };
  };
 };

//////////////////////////////////////////////////////////////////////////////
// main experimental trial loop starts here.
timeline.push(start_experiment);
for (let trial = 0; trial < triallist_length[2]; trial++) {
  // extend fixation for the first trial
  if (trial === 0){
    timeline.push(end_break_procedure);
    timeline.push(fixation_no_update);
  };
  if (break_trial_array.includes(trial) === true){
    timeline.push(start_break_procedure);
    timeline.push(end_break_procedure);
    timeline.push(fixation_no_update);  // extend fixation after break
  };
  timeline.push(fixation);
  timeline.push(gabor);
  timeline.push(percept_resp);
  timeline.push(conf_resp);
};

timeline.push(debriefing);
timeline.push(debriefing2);
timeline.push(endFullScreen);

// end of the whole experiment
////////////////////////////////////////////////////////////////////////////////

// run experiment
jsPsych.run(timeline);
