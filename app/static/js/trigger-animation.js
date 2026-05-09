var $animation_elements = $('.animated-element');
var $window = $(window);

function check_if_in_view() {
  var window_height = $window.height();
  var window_top_position = $window.scrollTop();
  var window_bottom_position = (window_top_position + window_height);
 
  $.each($animation_elements, function() {
    var $element = $(this);
    var element_height = $element.outerHeight();
    var element_top_position = $element.offset().top;
    var element_bottom_position = (element_top_position + element_height);

    //console.log($element);

    if ($('#python-logo').visible()) { 
        console.log("Viewed python logo");

    } else {
        console.log("Can't see python logo");
    }
 
    //check to see if this current container is within viewport
    if ((element_bottom_position >= window_top_position) &&
        (element_top_position <= window_bottom_position)) {
      //$element.addClass('in-view');
        //console.log('on-view');
      /* setTimeout(function(){
           $('#python-logo').addClass('animated bounceIn');
        }, 5000); */
    } else {
      //$element.removeClass('in-view');
      //console.log('not-on-view');
      /*$('#python-logo').removeClass('animated bounceIn');*/
    }
    

  });
}

$window.on('scroll resize', check_if_in_view);
$window.trigger('scroll');