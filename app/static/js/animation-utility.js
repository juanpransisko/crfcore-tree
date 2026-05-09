var $window = $(window);

function checkView() {

	if ($('#module-desc').visible(false)) {

		$('#python-logo').addClass('bounceIn');
		setTimeout(function () {
			$('#crf-logo').addClass('bounceIn');
		},300);
		setTimeout(function () {
			$('#tree-logo').addClass('bounceIn');
		},600);
		setTimeout(function () {
			$('#phil-logo').addClass('bounceIn');
		},900);
		
	} // if ($('#logos-wrapper').visible(false))

/*    if ( !($('#module-desc').visible(true)) ) {

    	$('#python-logo').removeClass('bounceIn');
        $('#crf-logo').removeClass('bounceIn');
        $('#tree-logo').removeClass('bounceIn');
        $('#phil-logo').removeClass('bounceIn');

    }*/ // if ( !($('#logos-wrapper').visible(true)) )

    if ($('.authors-img-1').visible(false)) {
    	$('.authors-img-1').css('filter','grayscale(0)');
    } else {
    	$('.authors-img-1').css('filter','grayscale(100%)');
    } //  if ($('.authors-img-1').visible(false))

    if ($('.authors-img-2').visible(false)) {
    	$('.authors-img-2').css('filter','grayscale(0)');
    } else {
    	$('.authors-img-2').css('filter','grayscale(100%)');
    } //  if ($('.authors-img-2').visible(false))


    if ($('#main-motivation').visible(false)) {
    	$('#main-motivation').css('opacity','1');
    	$('#motivation').css('opacity','1');
    }

    if ($('#system-archi').visible(true)) {
    	$('#system-archi').css('opacity','1');
    	$('#system-archi').addClass('animated zoomIn');
    }

    if ($('#see-docs').visible(false)) {
        $('#see-docs').css('opacity','1');
    }


} // function checkView()



/*$window.on('scroll resize', checkView);*/
$window.on('scroll', checkView);