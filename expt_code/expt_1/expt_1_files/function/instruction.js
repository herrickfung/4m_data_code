// contains all instruction text & survey text

var parti_info_questions = [
  [
    {
      type: 'text',
      prompt: "Prolific ID number: ",
      required: true,
    },
    {
      type: 'text',
      prompt: "Age: ",
      required: true,
    },
  ],

  [
    {
      type: 'drop-down',
      prompt: "Gender",
      options: ["Female", "Male", "Transgender", "Non-binary/Non-conforming", "Others", "Prefer not to say"],
      required: true,
    },
    {
      type: 'drop-down',
      prompt: "Ethnicity: ",
      options: ["Hispanic or Latino", "Not Hispanic or Latino", "Prefer not to say"],
      required: true,
    },
    {
      type: 'drop-down',
      prompt: "Race: ",
      options: ["American Indian / Alaska Native", "Asian", "Native Hawaiian or Other Pacific Islander", "Black or African American", "White", "More than one race", "Prefer not to say"],
      required: true,
    },
    {
      type: 'drop-down',
      prompt: "Vision: ",
      options: ["Normal", "Corrected-to-Normal (using glasses/contacts)", "Color-blind", "Others"],
      required: true,
    },
    {
      type: 'drop-down',
      prompt: "Handedness",
      options: ["Left-Handed", "Right-Handed", "Ambidextrous"],
      required: true,
    },
  ],
];

var welcome_text =[
  '<p style="font-size:20px;color:#000000;text-align:left">'+
  'Welcome to the experiment! Thank you for your participation.' +
  '<br> We are interested in how people make quick perceptual decisions.' +
  '<br><br> You may navigate back and forth with the buttons below or <br> with the Left and Right Arrow on your keyboard.' +
  '</p>',

  '<p style="font-size:20px;color:#000000;text-align:left">'+
   'We ask that you comply with the following experiment requirements:'+
  '<br><br> 1) In order for the experiment to run smoothly, stop any downloads or processes <br> that strain your internet connection.'+
  '<br><br> 2) Do not use your web browser\'s Back or Refresh buttons at any point during this experiment.'+
  '<br><br> 3) This experiment requires good concentration. As such, we ask that you complete the experiment <br> in an environment that is as free as possible of noise and distraction.'+
  '<br><br> Thank you for your cooperation.'+
  '</p>',

  '<p style="font-size:20px;color:#000000;text-align:left">'+
  'Most people complete this experiment in about 60 minutes. <br><br> You will complete 3 practice blocks, 2 calibration blocks, <br> and 5 runs of 6 experimental blocks, with each block containing 24 trials.' +
  '<br><br> You will have the opportunity to take break after each block.' +
  '<br><br> Please do your best to take the experiment in a single sitting <br> without excessive interruptions or taking very long breaks.'+
  '</p>',

  '<p style="font-size:20px;color:#000000;text-align:left">'+
  'The tasks in this experiment will involve discriminating whether an orientation patch is <br> tilted more to the left or more to the right, followed by giving a confidence judgement <br> on a scale from 1 to 4.' +
  '<br><br> More detailed task instruction will be given later.'+
  '<br><br> Now, we will switch to the fullscreen mode, please DO NOT escape the fullscreen mode <br> until the experiment will have been completed.'+
  '</p>',
];

var task_instruction = [
// On each trial, you will see quickly-presented orientation patches close to 45 degrees.
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'The following screens are the instruction of the task, please read carefully.' +
  '<br><br> On each trial, you will see a quickly-presented orientation patch close to 45 degrees.' +
  '<br><br> You may navigate back and forth with the buttons below or <br> with the Left and Right Arrow on your keyboard.' +
  '</p>',

  '<img src = "expt_1_files/img/gabor_45.png" width = 200px text-align = center>' +
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'This is a 45 degree patch.' +
  '<br><br> In the experiment, you will see patches that were tilted slightly off from this 45 degree,' +
  '<br> and you will have to judge whether it is tilted MORE LEFT or MORE RIGHT relative to this 45 degree.' +
  '</p>',

  '<table>' +
  '<tr>' +
  '<th>' +
  '<figure>' +
  '<img src = "expt_1_files/img/gabor_left.png" width = 200px text-align = center>' +
  '<figcaption> Tilted More Left than 45 </figcaption>' +
  '<figcaption> Press Left Arrow </figcaption>' +
  '<figcaption> on your keyboard </figcaption>' +
  '</th>' +
  '<th>' +
  '<figure>' +
  '<img src = "expt_1_files/img/gabor_45.png" width = 200px text-align = center>' +
  '<figcaption>  </figcaption>' +
  '<figcaption> 45 Degrees </figcaption>' +
  '<figcaption>  </figcaption>' +
  '</th>' +
  '<th>' +
  '<figure>' +
  '<img src = "expt_1_files/img/gabor_right.png" width = 200px text-align = center>' +
  '<figcaption> Tilted More Right than 45 </figcaption>' +
  '<figcaption> Press Right Arrow </figcaption>' +
  '<figcaption> on your keyboard </figcaption>' +
  '</th>' +
  '</tr>' +
  '</table>' +
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  "Here's an example." +
  '<br><br> The left figure is tilted more left than the center figure showing 45 degree,' +
  '<br><br> The right figure is tilted more right than the center figure,' +
  '<br><br> Use your right hand for this response.' +
  '</p>',

  '<img src = "expt_1_files/img/resp_screen.png" width = 300px text-align = center>' +
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'We made the previous example tilted off significantly. <br><br> However, in the actual experiment, the tilt changes will be a lot closer to 45 degrees.' +
  '<br><br> Thus, a white circle with a red line marked at the exact location of 45 degrees <br> will be presented whenever you need to make responses.' +
  '<br><br> You may use this as a reference to determine whether the stimulus is tilted more left or more right.' +
  '</p>',

  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'After the tilt response, indicate your confidence on whether your response is correct on a scale of 1-4:' +
  '<br><br> 1: very low confidence,' +
  '<br> 2: low confidence,' +
  '<br> 3 high confidence, and' +
  '<br> 4 very high confidence.' +
  '<br><br> Use the number on the left side of the keyboard and use your left hand for this response.'+
  '<br><br> Please make use of the WHOLE confidence scale to report your level of confidence faithfully.'+
  '</p>',

  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'Please go back and read the instruction if you are unclear about what to do.' +
  '<br><br> Otherwise, press next and get ready for the practice block!' +
  '</p>',
  ];

var start_prac_instruct_1 =
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'Make sure to <br><br>(1) Do your best, and <br>(2) Use the whole confidence scale!' +
  "<br><br> If you're ready, press spacebar to start the practice." +
  '</p>';

var start_prac_instruct_2 =
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'Great!' +
  '<br><br> Hopefully, the task is intuitive to you. In the previous practice block, we made the task ' +
  'quite easy. <br> The tilts in the actual experiment will be much closer to 45 degrees.' +
  '<br><br> In the following block, the tilts will progressively become closer to 45 degrees, <br>which will make the task progressively harder.' +
  '<br><br> Press spacebar now to start the second practice block.' +
  '</p>';

var start_prac_instruct_3 =
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'Well done!' +
  '<br><br> Unlike the previous block, where trials at the end became extremely hard.' +
  '<br> In the actual experiment, easy and difficult trials will be presented randomly.' +
  '<br><br> You will also NOT be receiving feedback for your responses in the actual experiment.' +
  '<br> To get used to not receiving feedback, this following practice block will give you no feedback.' +
  '<br><br> Press spacebar now to start the final practice block.'
  '</p>';

var start_staircase_instruct =
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'Great! You\'ve completed all practice blocks.' +
  '<br><br> We will now calibrate the stimulus for the rest of the experiment in the following two blocks.' +
  '<br><br> You DO NOT have to report confidence judgement for the following two blocks.' +
  '<br><br> The calibration may take longer than the previous practice blocks.' +
  '<br><br> Take a break before attempting.' +
  '<br><br> Try your best! <br><br> Press spacebar whenever you are ready.' +
  '</p>';

var staircase_break_instruct =
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'Well done! You\'ve completed the first calibration block.' +
  '<br><br> You may take a short break before pressing spacebar to continue with the second calibration block.'+
  '</p>';

var start_experiment_instruct =
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  'Cool! You\'ve completed all the calibration blocks, take a short break.' +
  '<br><br> You will then complete 5 runs of 6 experimental blocks,' +
  '<br> with each experimental block containing 24 trials.' +
  '<br><br> You may take breaks after each run and block.' +
  "<br><br> Don't be surprised if you see stimulus with different size, blurriness, and " +
  "<br> presentation time during the experiment, these are all part of the experiment."+
  '<br><br> Always make sure to — <br><br>(1) Do your best and <br>(2) Use the whole confidence scale!' +
  '<br><br> Press spacebar whenever you are ready for the experimental trials!' +
  '</p>';

var debrief_text =
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  "Excellent! You've completed the whole experiment." +
  "<p.>" +
  '<p style="font-size:26px;color:#FFFFFF;text-align:left">' +
  "DATA UPLOADING, DON'T CLOSE THE WINDOW!" +
  "<p.>" +
  '<p style="font-size:20px;color:#000000;text-align:left">' +
  "You will be redirected once it is completed." +
  "<br><br> You will also be prompt to save a csv file once it is completed, <br> you are strongly advised to keep a copy of the file just in case any technical issue occured." +
  "<br> You may delete the file permanently after you receive approval from the study." +
  "<br> The researchers will contact you only if they need you to send the file manually." +
  "<br> Otherwise, just wait until the approval happen, which will usually be within 24 hours." +
  "<br><br> Thank you for your participation, have a good day, and I hope to see you again!" +
  '</p>';


