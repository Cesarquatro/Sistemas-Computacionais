<?php
// Conecte-se ao banco de dados (ajuste as configurações conforme necessário)
$mysqli = new mysqli("127.0.0.1:1234", "root", "", "form");

// Verifique a conexão
if ($mysqli->connect_error) {
    die("Erro na conexão com o banco de dados: " . $mysqli->connect_error);
}


// Obtenha os dados do formulário
$email = $_POST['email'];
$telefone = $_POST['telefone'];
//$pass = $_POST['pass'];
$pass = password_hash($_POST['pass'], PASSWORD_DEFAULT);



// Insira os dados no banco de dados
$sql = "INSERT INTO bd (email, telefone, pass) VALUES (?, ?, ?)";
$stmt = $mysqli->prepare($sql);
$stmt->bind_param("sss", $email, $telefone,$pass);

if ($stmt->execute()) {
    echo "Dados inseridos com sucesso";
} else {
    echo "Erro ao inserir dados: " . $stmt->error;
}

// Feche a conexão
$stmt->close();
$mysqli->close();
?>