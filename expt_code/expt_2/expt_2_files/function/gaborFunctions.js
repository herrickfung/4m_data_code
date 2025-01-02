/*
* Make a sinusoidal grating. Creates a texture that later needs
* to be used with jglCreateTexture. Note: 0 deg means horizontal grating.
* If you want to ramp the grating with 2D Gaussian, also call function
* jglMakeGaussian and average the results of both functions.
*
* Parameters:
* width: in pixels
* height: in pixels
* numCyclesPer_stimSize: number of cycles present per stimSize number of pixels
* angle: in degrees
* phase: in degrees
*/
function jglMakeGrating(width, height, stimNumCycles, angle, phase) {

  // Convert angle and phase to radians
  angleInRad = ((angle + 0) * Math.PI) / 180;
  phaseInRad = (phase * Math.PI) * 180;

  // Get x and y coordinates for 2D grating
  xStep = 2 * Math.PI / width;
  yStep = 2 * Math.PI / height;
  x = jglMakeArray(-Math.PI, xStep, (Math.PI+.001));
  y = jglMakeArray(-Math.PI, yStep, (Math.PI+.001));

  // To tilt the 2D grating, we need to tilt x and y coordinates. These are tilting constants.
  xTilt = Math.cos(angleInRad) * stimNumCycles;
  yTilt = Math.sin(angleInRad) * stimNumCycles;

  //Create the grating
  var ixX, ixY; // x and y indices for arrays
  var grating = []; // 2D array
  for (ixX = 0; ixX < x.length; ixX++) {
    currentY = y[ixY];
    grating[ixX] = [];
    for (ixY = 0; ixY < y.length; ixY++) {
      grating[ixX][ixY] = Math.cos(x[ixX] * xTilt + y[ixY] * yTilt);
      // Scale to grayscale between 0 and 255
      grating[ixX][ixY] = Math.round(((grating[ixX][ixY] + 1) / 2) * 255);
    }
  }
  return grating;
}


/*
* Function to make array starting at low, going to high, stepping by step.
* Note: the last element is not "high" but high-step.
*
* Parameters:
* low: low bound of the array
* step: step between two elements of the array
* high: high bound of the array
*/
function jglMakeArray(low, step, high) {
  if (step === undefined) {
    step = 1;
  }
  var size = 0
  var array = []
  if (low < high) {
    size = Math.floor((high - low) / step);
    array = new Array(size);
    array[0] = low;
    for (var i = 1; i < array.length; i++) {
      array[i] = array[i - 1] + step;
    }
    return array;
  } else if (low > high) {
    size = Math.floor((low - high) / step);
    array = new Array(size);
    array[0] = low;
    for (var j = 1; j < array.length; j++) {
      array[j] = array[j - 1] - step;
    }
    return array;
  }
  return [low];
}


// change contrast
// this contrast is noise contrast range from 0 to 1,
// where 0 is no noise, 1 is many noise
function jglCreateGabor(ctx, array, mask, noise_contrast) {
  /* Note on how imageData's work.
  * ImageDatas are returned from createImageData,
  * they have an array called data. The data array is
  * a 1D array with 4 slots per pixel, R,G,B,Alpha. A
  * greyscale texture is created by making all RGB values
  * equal and Alpha = 255. The main job of this function
  * is to translate the given array into this data array.
  */
  if (!$.isArray(array)) {
    return;
  }
  var image;

  // 2D array passed in
  // var ctx = canvas.getContext('2d');
  image = ctx.createImageData(array.length, array.length);

  var row = 0;
  var col = 0;
  for (var i = 0; i < image.data.length; i += 4) {

    mask_val = mask[row][col]

    ran_val = Math.random() * 256;
    image.data[i + 0] = noise_contrast * ran_val + array[row][col] * (1-noise_contrast)
    image.data[i + 1] = noise_contrast * ran_val + array[row][col] * (1-noise_contrast)
    image.data[i + 2] = noise_contrast * ran_val + array[row][col] * (1-noise_contrast)
    image.data[i + 3] = mask_val

    col++;

    if (col == array[row].length) {
      col = 0;
      row++;
    }
  }

  return image;
}

/***********************************************
/* Make Gaussian Mask
/***********************************************/
function make2dMask(arr, amp, s) {
  var midX = Math.floor(arr.length / 2)
  var midY = Math.floor(arr[0].length / 2)
  var mask = []
  for (var i = 0; i < arr.length; i++) {
    var col = []
    for (var j = 0; j < arr[0].length; j++) {
      col.push(twoDGaussian(amp * 255, midX, midY, s, s, i, j))
    }
    mask.push(col)
  }
  return mask
}


function twoDGaussian(amplitude, x0, y0, sigmaX, sigmaY, x, y) {
  var exponent = -((Math.pow(x - x0, 2) / (2 * Math.pow(sigmaX, 2))) + (Math.pow(y - y0, 2) / (2 *
    Math.pow(sigmaY, 2))));
  return amplitude * Math.pow(Math.E, exponent);
}


function applyMask(arr, mask) {
  var masked_arr = []
  for (var i = 0; i < arr.length; i++) {
    var col = []
    for (var j = 0; j < arr[0].length; j++) {
      col.push(arr[i][j] * mask[i][j])
    }
    masked_arr.push(col)
  }
  return masked_arr
}
