$(document).ready(function() {

    //var SERVER_URI = "93.24.79.14:3333";
    var SERVER_URI = "92.154.62.215:3333";
    var IMAGE_URI = 'http://mo-ka.co/pi.php';
    var PRODUCT_IMAGE_URI = 'images/products/';

    var intervalID = undefined;
    var tIDs = [];
    var products = undefined;
    var socket = undefined;
    var totalPrice = 0;

    var nextTID = function() {
        return tIDs.length;
    };

    var setIntervalX = function (callback, delay, repetitions) {
        var x = 0;
        var intervalID = window.setInterval(function () {
           callback();
           if (++x === repetitions) {
               window.clearInterval(intervalID);
           }
        }, delay);
        return intervalID;
    };

    var refreshImage = function() {
        $('#image').attr('src', `${IMAGE_URI}?${new Date().getTime()}`);
    };

    var toggleStream = function() {
        if ($('.input-image').hasClass('streaming') && intervalID) {
            pauseStream();
        } else {
            playStream();
        }
    };

    var pauseStream = function() {
    	window.clearInterval(intervalID);
        $('.input-image').removeClass('streaming');
    };

    var playStream = function() {
		intervalID = setInterval(refreshImage, 1000);
        $('.input-image').addClass('streaming');
    };

    var addAnswer = function(pid) {
        var nTID = nextTID();
        tIDs.push(nTID);

        product = products.find(item => item.id === pid);
        if (product) {
            //$('#answers-list').prepend(`<li id="${nTID}-p">${answer}</li>`);
            $('#product-name').val('');

            var info = `
            - ${product.name}<br>
            - ${product.brand}<br>
            - ${product.weight}<br>
            - ${product.price} €
            `;

            var $img = $('#image').clone().removeAttr('id').removeClass().attr('id', `${nTID}-ti`);
            var $div = $(`<div class="timeline-image"><div class="timeline-image-info">${info}</div></div>`);
            $('#timeline').prepend($div.prepend($img));
            var text = 'article';
            if (tIDs.length > 1) text += 's';
            $('#scale-label').html(`${tIDs.length} ${text}`);

            totalPrice += product.price;
            $('#delay-label').html(`${totalPrice.toLocaleString()} €`);
        }

        $('#product-name').val('');
    };

    var handleProducts = function (data) {
        var product = undefined;
        for (var i = data[0].ids.length - 1; i >= 0; i--) {
            product = products.find(item => item.id === data[0].ids[i]);
            if (product) {
                var $productSuggestion = $(`#${product.id}-sgi`).remove();
                $('#suggestions').prepend($productSuggestion);
                $('#store-label').html(`${product.shelf}`);
            }
        }
    };


    // EVENTS

    $('#image').on('click', toggleStream);

    $('#product-name').keyup(function (e) {
        if (e.which == 13) { // Enter
            e.preventDefault();
            var answer = $('#product-name').val();
            if (answer) {
                addAnswer(answer);
            }
            playStream();
        } else {
        	if (!$('#product-name').val()) {
        		playStream();
        	} else {
        		pauseStream();
        	}
        }
    });

    $('#timeline').on('click', '.timeline-image', function() {
        $(this).addClass('selected').siblings().removeClass('selected');
        var id = `#${$(this).find('img').attr('id').split('-')[0]}-p`;
        $(id).addClass('selected').siblings().removeClass('selected');
    });

    $('#answers-list').on('click', 'li', function() {
        $(this).addClass('selected').siblings().removeClass('selected');
        $(`#${$(this).attr('id').split('-')[0]}-ti`).parent().addClass('selected').siblings().removeClass('selected');
    });

    $('#suggestions').on('click', '.suggestion-image', function() {
        var productId = $(this).attr('id').split('-')[0];
        addAnswer(productId);
    });

    $(document).keypress(function(e) {
        if (socket) {
            if (e.which == '114') { // r
                console.log('Sending validation...');
                socket.emit('user validation');
            }
        } else {
            console.error('Not connected to socket server.');
        }
    });
    
    // MAIN

    $.getJSON('products.json', function (json) {
        products = json;
        var source = [];
        for (var i = 0; i < products.length; i++) {
            source.push({label: `${products[i].brand} ${products[i].name}`, value: products[i].id});
        }
        $('#product-name').autocomplete({
            source: source,
            autoFocus: true
        });

        var html = '';
        for (var i = 0; i < products.length; i++) {
            html += `<div style="background-image: url('${PRODUCT_IMAGE_URI}${products[i].imageUrl}')" id="${products[i].id}-sgi" class="suggestion-image"></div>`
        }
        $('#suggestions').html(html);

        socket = io.connect(SERVER_URI, {'reconnection': false});

        socket.on('products', handleProducts);
        
        handleProducts([{ids:[products[Math.floor(Math.random() * products.length)].id, products[Math.floor(Math.random() * products.length)].id]}]);
    });

    //$('#product-name').focus();

    $('#image').trigger('click');

    //$('#scale-label').html('1,54 kilo');
    //$('#delay-label').html('(1m36s)');
    $('#user-name').html('Cl&eacute;mence Ariot');

});
