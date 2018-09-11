function Classes(o) {
	const color = ['#c00000', '#ffff00', '#00b0f0', '#92d050', '#ff0000', '#ffc000', '#002060', '#00b050', '#0070c0', '#7030a0'];
	
	const $classes = o.$classes;
	const names = o.names;
	const num = o.num || 10;
	const onChange = o.onChange;
	let currentIndex = 0;
	
	const classes = new Array();
	for (let i = 0; i < num; i++) {
		const name = names[i] || 'Class' + (i + 1);
		const linePaths = new Array();
		classes.push({
			name: name,
			color: color[i] || '#000000',
			linePaths: linePaths
		});
	}
	
	this.length = () => {
		return num;
	}
	
	this.name = (index = currentIndex) => {
		return classes[index].color.name;
	}
	
	this.names = () => {
		const ret = new Array();
		for (const i in classes) {
			ret.push(classes[i].name);
		}
		return ret;
	}
	
	this.color = (index = currentIndex) => {
		return classes[index].color;
	}
	
	this.getCurrentLinePaths = () => {
		return classes[currentIndex].linePaths;
	}
	
	this.setForm = () => {
		$classes.html('');
		const $classesDiv = $('<div />').appendTo($classes).append('<div>Classes: </div>');
		for (const i in classes) {
			const $classesSet = $('<span style="display: inline-block;" />').appendTo($classesDiv);
			const checked = (i == currentIndex) ? true : false;
			$('<input type="radio" name="classes" />').val(i).data('i', i).prop('checked', checked).appendTo($classesSet);
			$('<input type="text" name="classes" />').val(classes[i].name).attr('size', 10).data('i', i).appendTo($classesSet)
			.on('keyup', function() {
				classes[i].name = $(this).val();
			})
			.on('focus', function() {
				currentIndex = $(this).data('i');
				$classes.find('input[name="classes"][value="' + currentIndex + '"]').prop('checked', true);
				onChange();
			});
			$('<input type="color" />').val(color[i]).data('i', i).appendTo($classesSet)
			.on('click', function() {
				currentIndex = $(this).data('i');
				$classes.find('input[name="classes"][value="' + currentIndex + '"]').prop('checked', true);
				onChange();
			})
			.on('change', function() {
				classes[currentIndex].color = $(this).val();
				onChange();
			});
		}
		$('input[type="radio"][name="classes"]').on('change', function() {
			currentIndex = $(this).data('i');
			onChange();
		});
	}

	this.push = (lineWeight, points, index = currentIndex) => {
		classes[index].linePaths.push({
			lineWeight: lineWeight,
			points: points
		});
	}
	
	this.pop = (index = currentIndex) => {
		classes[index].linePaths.pop();
	}
	
	this.getColorData = ($canvas) => {
		const width = $canvas.get(0).width, height = $canvas.get(0).height;

		const imgContext = $canvas.get(0).getContext('2d');
		const imageData = imgContext.getImageData(0, 0, width, height);
		
		const ret = new Array();
		for (let i = 0; i < this.length(); i++) {
			const $pointsCanvas = $('<canvas />').appendTo('body').hide();
			$pointsCanvas.get(0).width = width;
			$pointsCanvas.get(0).height = height;
			const pointsContext = $pointsCanvas.get(0).getContext('2d');
	
			this.drawLinePaths($pointsCanvas, i);
			const pointsData = pointsContext.getImageData(0, 0, width, height);
			const data = pointsData.data;
	
			for (let j = 0; j < data.length; j += 4) {
				if (data[j + 3] > 0) { //alphaチャンネルに値があれば
					const coords = {x: (j / 4) % width, y: Math.floor((j / 4) / width)};
					const coord = {x: (j / 4) % width, y: Math.floor((j / 4) / width)};
					const rgb = {r: imageData.data[j], g: imageData.data[j + 1], b: imageData.data[j + 2]};
					const lab = rgb2lab(rgb);
					const hsv = rgb2hsv(rgb);
					ret.push({
						Class: i,
						x: coords.x,
						y: coords.y,
						B: rgb.b,
						G: rgb.g,
						R: rgb.r,
						H: hsv.h,
						S: hsv.s,
						V: hsv.v,
						L: lab.l,
						a: lab.a,
						b: lab.b
					});
				}
	    	}
			$pointsCanvas.remove();
		}
		return ret;
	}

	this.drawLinePaths = ($canvas, index = currentIndex) => {
		const linePaths = classes[index].linePaths;
		const c = $canvas.get(0).getContext('2d');
		c.clearRect(0, 0, $canvas.width(), $canvas.height());
		for (let i = 0; i < linePaths.length; i++) {
			const lineWeight = linePaths[i].lineWeight;
			if (lineWeight == 0) {//郭さん提供ソースのアルゴリズム（デバッグ用）
				c.fillStyle = classes[index].color;
				for (let j = 0; j < linePaths[i].points.length; j++) {
					c.fillRect(linePaths[i].points[j].x, linePaths[i].points[j].y, 2, 2);
				}
			}
			else {
				c.beginPath();
				c.strokeStyle = classes[index].color;
				c.lineCap = "round";
				c.lineWidth = lineWeight;
				c.moveTo(linePaths[i].points[0].x, linePaths[i].points[0].y);
				for (let j = 1; j < linePaths[i].points.length; j++) {
					c.lineTo(linePaths[i].points[j].x, linePaths[i].points[j].y);
				}
				c.stroke();				
			}
		}
	}
	
}

function createTrainingData(files) {
	const dataBuffer = new Array();
	
	$(window).on('resize', (e) => {
		canvasResize();
		redraw();
	});
	
	//filesをセット
	const $filesDiv = $('<div />').appendTo('#files').append('<span>Files: </span>');
	const $filesSelect = $('<select id="file" />').appendTo($filesDiv);	
	for (let i = 0; i < files.length; i++) {
		const file = files[i];
		if (!file || file.type.indexOf('image/') < 0) continue; //画像以外は無視
		const count = dataBuffer.length;
		dataBuffer.push({
			file: file,
			image: undefined, //後からセット
			offset: {x: 0, y: 0},
			scale: 1,
			classes: new Classes({
				$classes: $('#classes'),
				names: ['Class1', 'Class2'],
				num: 10,
				onChange: redrawPathCanvas
			})
		});
		$('<option />').val(count).html(file.name).data('i', count).appendTo($filesSelect);
	}
	
	setData();
	$filesSelect.on('change', (e) => {
		setData();
	});
	
	//LineWeight選択<select>
	const $lineWeightDiv = $('<div />').appendTo('#control').append('<span>LineWeight: </span>');
	const $lineWeightselect = $('<select id="lineWeight" />').appendTo($lineWeightDiv);
	for (const i of [1, 2, 3, 5, 7, 10, 15, 20, 30]) {
		$('<option />')
		.val(i)
		.html(i)
		.appendTo($lineWeightselect);
	}
	//初期値に5をセット
	$lineWeightselect.val(5);
	//郭さん提供ソースのアルゴリズム（デバッグ用）
	$lineWeightselect.append('<option value=0>郭さん提供ソースのアルゴリズム</option>');
	
	//Undoボタン
	const $undoButton = $('<button>Undo</button>').appendTo('#control');
	$undoButton.on('click', (e) => {
		dataBuffer[$('#file').val()].classes.pop();
		redrawPathCanvas();
	});
	
	//Saveボタン	
	const $saveButton = $('<button>Save</button>').appendTo('#control');
	$saveButton.on('click', (e) => {
		const lines = new Array();
		for (let i = 0; i < dataBuffer.length; i++) {
			const buf = dataBuffer[i];
			const classNames = buf.classes.names();
			const fileName = buf.file.name;
			if (buf.image === undefined) continue;
			const colorData = buf.classes.getColorData($('#offCanvas'));
			for (const j in colorData) {
				lines.push(`${classNames[colorData[j].Class]},${fileName},${colorData[j].x},${colorData[j].y},${colorData[j].B},${colorData[j].G},${colorData[j].R},${colorData[j].H},${colorData[j].S},${colorData[j].V},${colorData[j].L},${colorData[j].a},${colorData[j].b}`);
			}
		}
		//ダウンロード処理
    	fileDownload('Class,Image,x,y,B,G,R,H,S,V,L,a,b\n' + lines.join('\n'), 'text/csv', 'traningData.csv');
	});
	
	function setData() {
		const fileIndex = $('#file').val();
		
		dataBuffer[fileIndex].classes.setForm();

		//image読み込み
		if (dataBuffer[fileIndex].image) {
			createCanvas();
			redrawPathCanvas();
		}
		else {
			const reader = new FileReader();
			reader.readAsDataURL(dataBuffer[fileIndex].file);
			reader.onload = (e) => {
				const image = new Image();
				image.src = e.target.result;
				image.onload = () => {
					dataBuffer[fileIndex].image = image;
					//canvas設定
					createCanvas();
					redrawPathCanvas();
				}
			}
		}
		
	}
	
	function createCanvas() {
		const scaleMaxRatio = 5;
		const currentFile = dataBuffer[$('#file').val()];
		
		//canvasエリア初期化
		$('#canvasArea').html('');
		
		//canvas設置
		const $canvas = $('<canvas id="canvas" />').appendTo('#canvasArea');
		const context = $canvas.get(0).getContext('2d');
		canvasResize();
		const width = currentFile.image.width, height = currentFile.image.height;

		//pathCanvas設置
		const $pathCanvas = $('<canvas id="pathCanvas" />').appendTo('#canvasArea').hide();
		const pathCanvasContext = $pathCanvas.get(0).getContext('2d');
		$pathCanvas.get(0).width = width;
		$pathCanvas.get(0).height = height;

		//canvas描画
		redraw();
		
		//offCanvas設置（オリジナル画像バッファ用）
		const $offCanvas = $('<canvas id="offCanvas" />').appendTo('#canvasArea').hide();
		const offCanvasContext = $offCanvas.get(0).getContext('2d');
		$offCanvas.get(0).width = width;
		$offCanvas.get(0).height = height;
		offCanvasContext.drawImage(currentFile.image, 0, 0, width, height);
	
		//スライダー設置
		const $sliderDiv = $('<div />').appendTo('#canvasArea');
		const $slider = $('<input type="range" min="0.1" max="' + scaleMaxRatio + '" step="any" />').val(currentFile.scale).appendTo($sliderDiv);
		$slider.on('input change', function() {
			const value = $(this).val();
			currentFile.scale = value;
			$('#sliderVal').html('×' + (Math.round(value * 10) / 10));
			redraw();
		});
		$('<span id="sliderVal">×1.0</span>').appendTo($sliderDiv);
	
		//canvas上でのマウスイベント
		const points = {
			canvas: new Array(), //メインcanvas上での座標
			actual: new Array() //オフcanvas上での座標（実際の画像上での座標）
		};
		const mouse = {
			pressed: -1, //マウスボタン押下、-1:押下なし、0:左ボタン押下中、1:中ボタン押下中、2:右ボタン押下中
			start: {x: 0, y: 0} //左ボタンが押下されたポイントを記録
		};
		const initialOffset = {x: 0, y: 0}; //左ボタンが押された時のOffsetを記録	
		
		$canvas
		.on('contextmenu', (e) => {
			e.preventDefault();
		})
		.on('mousedown', (e) => {
			if (mouse.pressed != -1) return; //一つのボタンが押下中に別のボタンが押下された時は無視する
			mouse.pressed = e.button;
			const rect = e.target.getBoundingClientRect();
			mouse.start.x = ~~(e.clientX - rect.left);
	        mouse.start.y = ~~(e.clientY - rect.top);
	        initialOffset.x = currentFile.offset.x;
	        initialOffset.y = currentFile.offset.y;
			if (mouse.pressed == 2) {
				const actualCoord = {
					x: (mouse.start.x - currentFile.offset.x) / currentFile.scale,
					y: (mouse.start.y - currentFile.offset.y) / currentFile.scale
				};
				//座標を格納
				points.canvas.length = 0;
				points.canvas.push({x: mouse.start.x, y: mouse.start.y});
				points.actual.length = 0;
				points.actual.push(actualCoord);
			}
		})
		.on('mousemove', (e) => {
			if (mouse.pressed == -1) return;
			const rect = e.target.getBoundingClientRect();
			const x = ~~(e.clientX - rect.left);
			const y = ~~(e.clientY - rect.top);
			if (mouse.pressed == 0) {
				currentFile.offset.x = x - mouse.start.x + initialOffset.x;
				currentFile.offset.y = y - mouse.start.y + initialOffset.y;
				redraw();
			}
			else if (mouse.pressed == 2) {
				const actualCoord = {
					x: (x - currentFile.offset.x) / currentFile.scale,
					y: (y - currentFile.offset.y) / currentFile.scale
				};
				//描画開始
				const lineWeight = $('#lineWeight').val();
				//メインcanvas
				if (lineWeight == 0) { //郭さん提供ソースのアルゴリズム（デバッグ用）
					context.fillStyle = "rgb(255, 255, 255)";
					context.fillRect(x, y, 2, 2);
				}
				else {
					context.beginPath();
					context.strokeStyle = "rgb(255, 255, 255)";
					context.lineCap = "round";
					context.lineWidth = lineWeight * currentFile.scale;
					context.moveTo(points.canvas[points.canvas.length - 1].x, points.canvas[points.canvas.length - 1].y);
					context.lineTo(x, y);
					context.stroke();
				}
				//オフcanvas
				if (lineWeight == 0) { //郭さん提供ソースのアルゴリズム（デバッグ用）
					pathCanvasContext.fillStyle = currentFile.classes.color();
					pathCanvasContext.fillRect(actualCoord.x, actualCoord.y, 2, 2);
				}
				else {
					pathCanvasContext.beginPath();
					pathCanvasContext.strokeStyle = currentFile.classes.color();
					pathCanvasContext.lineCap = "round";
					pathCanvasContext.lineWidth = lineWeight;
					pathCanvasContext.moveTo(points.actual[points.actual.length - 1].x, points.actual[points.actual.length - 1].y);
					pathCanvasContext.lineTo(actualCoord.x, actualCoord.y);
					pathCanvasContext.stroke();
				}
				
				//座標を格納
				points.canvas.push({x: x, y: y});
				points.actual.push(actualCoord);
			}
		})
		.on('mouseup', (e) => {
			if (mouse.pressed == e.button) {
				const rect = e.target.getBoundingClientRect();
				const x = ~~(e.clientX - rect.left);
				const y = ~~(e.clientY - rect.top);
				if (mouse.pressed == 0) {
					currentFile.offset.x = x - mouse.start.x + initialOffset.x;
					currentFile.offset.y = y - mouse.start.y + initialOffset.y;
				}
				else if (mouse.pressed == 2) {
					redraw();
					dataBuffer[$('#file').val()].classes.push($('#lineWeight').val(), points.actual.slice());
				}
				mouse.pressed = -1;
			}
		})
		.on('mouseout', (e) => {
			if (mouse.pressed == 2) {
				redraw();
				dataBuffer[$('#file').val()].classes.push($('#lineWeight').val(), points.actual.slice());
			}
			mouse.pressed = -1;
		});
	
		//canvas上でのホイール動作
		const mousewheelevent = 'onwheel' in document ? 'wheel' : 'onmousewheel' in document ? 'mousewheel' : 'DOMMouseScroll';
		$canvas.on(mousewheelevent, (e) => {
			e.preventDefault();
			if (mouse.pressed != -1) return; //マウスボタン押下中は無視する
			const delta = e.originalEvent.deltaY ? -(e.originalEvent.deltaY) : e.originalEvent.wheelDelta ? e.originalEvent.wheelDelta : -(e.originalEvent.detail);
			const newScale = (delta > 0) ? Math.min(currentFile.scale * 1.1, scaleMaxRatio) : Math.max(currentFile.scale * 0.9, 0.1);
	
			const pointerOffsetX = e.clientX - $canvas.offset().left;
			const pointerOffsetY = e.clientY - $canvas.offset().top;
			
			currentFile.offset.x = pointerOffsetX - (pointerOffsetX - currentFile.offset.x) * (newScale / currentFile.scale);
			currentFile.offset.y = pointerOffsetY - (pointerOffsetY - currentFile.offset.y) * (newScale / currentFile.scale);
	
			currentFile.scale = newScale;
			$slider.val(currentFile.scale);
			$('#sliderVal').html('×' + (Math.round(currentFile.scale * 10) / 10));
			redraw();
		});
	}
	
	function fileDownload(content, mimeType, name) {
		const bom = new Uint8Array([0xEF, 0xBB, 0xBF]);
		const blob = new Blob([bom, content], {type : mimeType});

		const a = document.createElement('a');
		a.download = name;
		a.target   = '_blank';
		
		if (window.navigator.msSaveBlob) {
			// for IE
			window.navigator.msSaveBlob(blob, name)
		}
		else if (window.URL && window.URL.createObjectURL) {
			// for Firefox
			a.href = window.URL.createObjectURL(blob);
			document.body.appendChild(a);
			a.click();
			document.body.removeChild(a);
		}
		else if (window.webkitURL && window.webkitURL.createObject) {
			// for Chrome
			a.href = window.webkitURL.createObjectURL(blob);
			a.click();
		}
		else {
			// for Safari
			window.open('data:' + mimeType + ';base64,' + window.Base64.encode(content), '_blank');
		}
	}
	
	function redraw() {
		const currentFile = dataBuffer[$('#file').val()];
		const $canvas = $('#canvas');
		const c = $canvas.get(0).getContext('2d');
		c.clearRect(0, 0, $canvas.width(), $canvas.height());
		c.fillStyle = 'gray';
		c.fillRect(0, 0, $canvas.width(), $canvas.height());
		c.drawImage(currentFile.image, currentFile.offset.x, currentFile.offset.y, currentFile.image.width * currentFile.scale, currentFile.image.height * currentFile.scale);
		const $pathCanvas = $('#pathCanvas');
    	c.drawImage($pathCanvas.get(0), currentFile.offset.x, currentFile.offset.y, currentFile.image.width * currentFile.scale, currentFile.image.height * currentFile.scale);
	}
	
	function redrawPathCanvas() {
		dataBuffer[$('#file').val()].classes.drawLinePaths($('#pathCanvas'));
		redraw();
	}

	function canvasResize() {
		const width = $(window).width() - 100;
		const height = $(window).height() - 300;
		const $canvas = $('#canvas');
		$canvas.get(0).width = width;
		$canvas.get(0).height = height;
	}
	
}


	
