var loadPage = false;
var lastUrl = '';

function changeLanguage(language, css) {
	lang = language;
	var href = window.location.pathname + window.location.search;
	var urlParts = href.split("/");
	if (urlParts[0] == '')
		urlParts.shift();
	if (urlParts[0].length == 2) {
		urlParts.shift();
	}
	var url = '/' + language;
	if (urlParts.length == 1 && urlParts[0] == "") {
		url += "/index";
	}
	else {
		jQuery.each(urlParts, function(index, value) {
			if (value != '')
				url += '/' + value;
		});
	}
	if (css != '') {
		$('#mainNav .lang span').addClass('hidden');
		$('#mainNav .lang span.' + css).removeClass('hidden');
		$('#mainNav .lang ul li span').removeClass('hidden');
		$('#mainNav .lang ul li span.' + css).addClass('hidden');
	}
	window.location.href = url;
}

function getMainContent(href) {
	loadPage = true;
	if (lang == undefined || lang == '') {
		lang = 'hu';
	}
	var urlParts = href.split("/");
	if (urlParts[0] == '')
		urlParts.shift();
	if (urlParts[0].length == 2) {
		urlParts.shift();
	}
	var url = '/' + lang;
	var home = true;
	jQuery.each(urlParts, function(index, value) {
		if (value != '') {
			url += '/' + value;
			home = false;
		}
	});

	if (home) {
		url = '/';
	}

	$.get(url, function(data){
		//set url
		if (data['pageData']) {
			var seo = data['seo'];
			if (url != lastUrl) {
				if (typeof (window.history.pushState) == 'function') {
					window.history.pushState(new Object(), seo['title'], url);
				}
			}
			var pageData = data['pageData'];
			jQuery.each(pageData, function(index, value) {
				jQuery(value.area).html(value.content);
				bindHrefClickEvents(value.area);
			});
			document.title = seo['title'];
			if (seo['keywords'] != '') {
				var mt = $('meta[name=keywords]');
				mt = mt.length ? mt : $('<meta name="keywords" />').appendTo('head');
				mt.attr('content', seo['keywords']);
			}
			if (seo['description'] != '') {
				var mt = $('meta[name=description]');
				mt = mt.length ? mt : $('<meta name="description" />').appendTo('head');
				mt.attr('content', seo['description']);
			}
			scrollToElement('.mainWrapCols', -81);
	    	//$.scrollTo( '.mainWrapCols', 800, {offset:-81});
			if(urlParts[0]=='news'){
				initNews();
			}
		}
		else {
	    	$('#main .rightCol').html(data['content']);
	    	bindHrefClickEvents('#main .rightCol');
			scrollToElement('.mainWrapCols', -81);
	    	//$.scrollTo( '.mainWrapCols', 800, {offset:-81});
		}
		lastUrl = url;
		loadPage = false;
    }, "json");
}

function getRCMainContent(href) {

	$.get(href, function(data){
    	$('#main .rightCol').html(data);
    	bindHrefClickEvents('#main .rightCol');
    	$.scrollTo( '.mainWrapCols', 800, {offset:-81});
	});
}

function checkURL()
{
	if (loadPage == false && typeof (window.history.pushState) == 'function') {
		var href = window.location.pathname + window.location.search;    //if no parameter is provided, use the hash value from the current address
	    if(href != lastUrl) // if the hash value has changed
	    {
	    	if (lastUrl != undefined && lastUrl != '') {
	    		lastUrl=href;
	    		getMainContent(href, true); // and load the new page
	    	}
	    	lastUrl=href;   //update the current hash
    	}
    }
}

function checkLocation() {



	var urlParts = document.location.pathname.split("/");
	if (urlParts[0] == '')
		urlParts.shift();
	if (urlParts[0].length == 2) {
		urlParts.shift();
	}

	if (urlParts[0] == '' || document.location.pathname === '/'){
		return '/';
	}

	return urlParts[0];
};