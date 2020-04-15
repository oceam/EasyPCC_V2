//参照
//OpenCV: Color conversions
//https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html

function rgb2hsv(rgb) {
	var r = rgb.r / 255;
	var g = rgb.g / 255;
	var b = rgb.b / 255;

	var v = Math.max(r, g, b);
	var min = Math.min(r, g, b);
	var s = (v == 0) ? 0 : (v - min) / v;
	
	var h = 0;
	if (v == r) {
		h = 60 * (g -b) / (v - min);
	}
	else if (v == g) {
		h = 120 + 60 * (b - r) / (v - min);
	}
	else if (v == b) {
		h = 240 + 60 * (r - g) / (v - min);
	}
	if (h < 0) h += 360;

	v *= 255;
	s *= 255;
	h /= 2;

	return {
		h: Math.round(h),
		s: Math.round(s),
		v: Math.round(v)
	};
}

function rgb2lab(rgb) {
    var r = rgb.r / 255;
    var g = rgb.g / 255;
    var b = rgb.b / 255;

	//converted to the floating-point format and scaled to fit the 0 to 1 range
	r = (r > 0.04045) ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
	g = (g > 0.04045) ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
	b = (b > 0.04045) ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

	var x = 0.412453 * r + 0.357580 * g + 0.180423 * b;
	var y = 0.212671 * r + 0.715160 * g + 0.072169 * b;
	var z = 0.019334 * r + 0.119193 * g + 0.950227 * b;
	
	x /= 0.950456;
	z /= 1.088754;
	var l = (y > 0.008856) ? 116 * Math.pow(y, (1/3)) - 16 : 903.3 * y;

	var fx = (x > 0.008856) ? Math.pow(x, (1/3)) : (7.787 * x) + (16/116);
	var fy = (y > 0.008856) ? Math.pow(y, (1/3)) : (7.787 * y) + (16/116);
	var fz = (z > 0.008856) ? Math.pow(z, (1/3)) : (7.787 * z) + (16/116);

	var a = 500 * (fx - fy);
    var b = 200 * (fy - fz);
	
    return {
    	l: Math.round(l * 255 /100),
    	a: Math.round(a + 128),
    	b: Math.round(b + 128)
    };
}
