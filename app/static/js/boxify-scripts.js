//$(document).ready(function() {

	/***************** Waypoints ******************/

/*	$('.wp1').waypoint(function() {
		$('.wp1').addClass('animated fadeInLeft');
	}, {
		offset: '75%'
	});
	$('.wp2').waypoint(function() {
		$('.wp2').addClass('animated fadeInDown');
	}, {
		offset: '75%'
	});
	$('.wp3').waypoint(function() {
		$('.wp3').addClass('animated bounceInDown');
	}, {
		offset: '75%'
	});
	$('.wp4').waypoint(function() {
		$('.wp4').addClass('animated fadeInDown');
	}, {
		offset: '75%'
	});*/

	/***************** Flickity ******************/

/*	$('#featuresSlider').flickity({
		cellAlign: 'left',
		contain: true,
		prevNextButtons: false
	});

	$('#showcaseSlider').flickity({
		cellAlign: 'left',
		contain: true,
		prevNextButtons: false,
		imagesLoaded: true
	});
*/
	/***************** Fancybox ******************/

	/*$(".youtube-media").on("click", function(e) {
		var jWindow = $(window).width();
		if (jWindow <= 768) {
			return;
		}
		$.fancybox({
			href: this.href,
			padding: 4,
			type: "iframe",
			'href': this.href.replace(new RegExp("watch\\?v=", "i"), 'v/'),
		});
		return false;
	});*/

//});

/*$(document).ready(function() {
	$("a.single_image").fancybox({
		padding: 4,
	});
});*/

/***************** Nav Transformicon ******************/

/* When user clicks the Icon */
$(".nav-toggle").click(function() {
	$(this).toggleClass("active");
	$(".overlay-boxify").toggleClass("open");
	if ( $(this).hasClass("active") ) {
		//console.log('Active');
		document.getElementById("header-background").style.backgroundColor = 'transparent';
		//document.getElementById("header-background").style.boxShadow = 'none';
		disableScrolling();
		//window.onscroll = function () { window.scrollTo(0, 0); };
	} else {
		//console.log('Not Active');
		if (document.body.scrollTop > 100 || document.documentElement.scrollTop > 100) {
    		document.getElementById("header-background").style.backgroundColor = '#006341';
    		//document.getElementById("header-background").style.boxShadow = '0 2px 5px grey';
	    } else {
	    	document.getElementById("header-background").style.backgroundColor = 'transparent';
	    	//document.getElementById("header-background").style.boxShadow = 'none';
	    }
	    enableScrolling();
	}
});

/* When user clicks a link */
$(".overlay ul li a").click(function() {
	$(".nav-toggle").toggleClass("active");
	$(".overlay-boxify").toggleClass("open");
});

/* When user clicks outside */
$(".overlay").click(function() {
	$(".nav-toggle").toggleClass("active");
	$(".overlay-boxify").toggleClass("open");
	
	if (document.body.scrollTop > 100 || document.documentElement.scrollTop > 100) {
    	document.getElementById("header-background").style.backgroundColor = '#006341';
    	//document.getElementById("header-background").style.boxShadow = '0 2px 5px grey';
    } else {
    	document.getElementById("header-background").style.backgroundColor = 'transparent';
    	//document.getElementById("header-background").style.boxShadow = 'none';
    }

    enableScrolling();

});

function disableScrolling(){
    var x=window.scrollX;
    var y=window.scrollY;
    window.onscroll=function(){window.scrollTo(x, y);};
}

function enableScrolling(){
    window.onscroll = function() {scrollFunction()};

}

/***************** Smooth Scrolling ******************/

/*$('a[href*=#]:not([href=#])').click(function() {
	if (location.pathname.replace(/^\//, '') === this.pathname.replace(/^\//, '') && location.hostname === this.hostname) {

		var target = $(this.hash);
		target = target.length ? target : $('[name=' + this.hash.slice(1) + ']');
		if (target.length) {
			$('html,body').animate({
				scrollTop: target.offset().top
			}, 2000);
			return false;
		}
	}
});
*/


/* Button Ripple Efects */

(function (window, $) {
  
  $(function() {
    
    $('.ripple').on('click', function (event) {
      event.preventDefault();
      
      var $div = $('<div/>'),
          btnOffset = $(this).offset(),
      		xPos = event.pageX - btnOffset.left,
      		yPos = event.pageY - btnOffset.top;
      
      $div.addClass('ripple-effect');
      var $ripple = $(".ripple-effect");
      
      $ripple.css("height", $(this).height());
      $ripple.css("width", $(this).height());
      $div
        .css({
          top: yPos - ($ripple.height()/2),
          left: xPos - ($ripple.width()/2),
          background: $(this).data("ripple-color")
        }) 
        .appendTo($(this));

      window.setTimeout(function(){
        $div.remove();
      }, 2000);
    });
    
  });
  
})(window, jQuery);


/* Progress Button */

/*$('.btn-submit').on('click', function() {

	$('.loading-spinner').css('display','inline-block');

	setTimeout(function() {
		$('.loading-spinner').css('display','none');
		$('.done-marker').css('display','inline-block');

	   	setTimeout(function() {
		   $('.done-marker').css('display','none');
		}, 3000);

	}, 5000);


});*/
