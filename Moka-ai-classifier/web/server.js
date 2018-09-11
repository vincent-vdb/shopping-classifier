//var mysql = require('mysql')
var io = require('socket.io').listen(3333);

// Define our db creds
// var db = mysql.createConnection({
//     host: 'localhost',
//     user: 'root',
//     database: 'node'
// })
 
// Log any errors connected to the db
// db.connect(function(err){
//     if (err) console.log(err)
// })
 
var products = [];
var isInitProducts = false;
var socketCount = 0;
var maxProducts = 1;
 
io.sockets.on('connection', function(socket){
    console.log('Node server: new connection from ' + socket.request.connection.remoteAddress);

    // Socket has connected, increase socket count
    socketCount++;
    // Let all sockets know how many are connected
    io.sockets.emit('users connected', socketCount);
 
    socket.on('disconnect', function() {
        // Decrease the socket count on a disconnect, emit
        socketCount--;
        io.sockets.emit('users connected', socketCount);
    })
 
    socket.on('product found', function(data){
        products.push(data);
        if (products.length > maxProducts) products.shift();
        io.sockets.emit('products', products);
        // Use node's db injection format to filter incoming data
        //db.query('INSERT INTO notes (note) VALUES (?)', data.note)
    });

    socket.on('user validation', function(data){
        console.log('Node server: user validation');
        io.sockets.emit('validate');
        // Use node's db injection format to filter incoming data
        //db.query('INSERT INTO notes (note) VALUES (?)', data.note)
    });
 
    // Check to see if initial query/notes are set
    if (!isInitProducts) {
        // Initial app start, run db query
        // db.query('SELECT * FROM notes')
        //     .on('result', function(data){
        //         // Push results onto the notes array
        //         notes.push(data)
        //     })
        //     .on('end', function(){
        //         // Only emit notes after query has been completed
        //         socket.emit('initial notes', notes)
        //     })
 
        isInitProducts = true;
    }
    socket.emit('products', products);
});
