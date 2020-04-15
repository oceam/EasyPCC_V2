$(function() {

	//document全体へのファイルドロップを禁止する
	$(document)
	.on('dragover', function(e){
		e.stopPropagation();
    	e.preventDefault();
	})
	.on('drop', function(e){
		e.stopPropagation();
    	e.preventDefault();
	});

	//$fileInput（非表示）、下記の$dropAreaのclickで発火
	$fileInput = $('<input type="file" accept="image/*" style="display: none" multiple="multiple">')
	.on('change', function(e){
		$dropArea.remove();
		createTrainingData(e.target.files);
		$(this).val = '';
	})
	.appendTo('#files');
	
	//dropArea作成
	$dropArea = $('<div id="dropArea">Drop files here or click to select files.</div>')
	.on('dragover', function(e){
		e.stopPropagation();
        e.preventDefault();
        $(this).addClass('dragover');
	})
	.on('dragleave', function(e){
		$(this).removeClass('dragover');
	})
	.on('drop', function(e){
		e.preventDefault();
		$(this).removeClass('dragover');
		const files = e.originalEvent.dataTransfer.files;
		$dropArea.remove();
		createTrainingData(files);
	})
	.on('click', function(e){
		$fileInput.click();
	})
	.appendTo('#files');

});