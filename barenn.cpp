#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;

class DenseLayer {
    int in_features;
    int out_features;
    bool final;
    // pesos contíguos e transpostos: weights[j * out_features + i] (entrada j, neurônio i)
    vector<float> weights;
    vector<float> bias;
    // Dados do último forward usados para calcular os gradientes.
    const float* last_input = nullptr;
    vector<float> last_z;
    // buffers reaproveitados entre amostras
    vector<float> out;
    vector<float> delta;
    vector<float> grad_in;

public:
    DenseLayer(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(static_cast<size_t>(input_size) * output_size), bias(output_size, 0),
          last_z(output_size), out(output_size), delta(output_size),
          grad_in(input_size) {
        // A semente fixa torna a inicialização de cada camada reproduzível.
        // mesma ordem de sorteio do legado
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                weights[j * out_features + i] = distribution(gen);
            }
        }
    }

    const float* Forward(const float* input) {
        last_input = input;
        float* z = last_z.data();
        for (int i = 0; i < out_features; ++i) {
            z[i] = bias[i];
        }
        // mesma ordem de soma do legado (j = 0, 1, ...)
        for (int j = 0; j < in_features; ++j) {
            const float x = input[j];
            const float* w = &weights[j * out_features];
            for (int i = 0; i < out_features; ++i) {
                z[i] += x * w[i];
            }
        }
        // As camadas ocultas usam ReLU. A última devolve escores lineares.
        for (int i = 0; i < out_features; ++i) {
            out[i] = final ? z[i] : max(0.0f, z[i]);
        }
        return out.data();
    }

    // a primeira camada não precisa do gradiente da entrada
    const float* Backward(const float* grad_out, float learning_rate,
                          bool propagate = true) {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        for (int i = 0; i < out_features; ++i) {
            delta[i] = final ? grad_out[i]
                             : grad_out[i] * (last_z[i] > 0.0f ? 1.0f : 0.0f);
        }

        // O gradiente da entrada usa os pesos anteriores à atualização.
        if (propagate) {
            for (int i = 0; i < in_features; ++i) {
                const float* w = &weights[i * out_features];
                float sum = 0;
                for (int j = 0; j < out_features; ++j) {
                    sum += delta[j] * w[j];
                }
                grad_in[i] = sum;
            }
        }

        // SGD por amostra. O gradiente do peso é delta vezes a entrada.
        // O gradiente do viés é o próprio delta.
        for (int j = 0; j < in_features; ++j) {
            const float x = last_input[j];
            float* w = &weights[j * out_features];
            for (int i = 0; i < out_features; ++i) {
                w[i] -= learning_rate * (delta[i] * x);
            }
        }
        for (int i = 0; i < out_features; ++i) {
            bias[i] -= learning_rate * delta[i];
        }
        return propagate ? grad_in.data() : nullptr;
    }

    void compartilhar_parametros() {
        MPI_Bcast(weights.data(), static_cast<int>(weights.size()), MPI_FLOAT,
                  0, MPI_COMM_WORLD);
        MPI_Bcast(bias.data(), out_features, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
};

// imagens em um vetor só: a imagem i começa em i * entradas
struct Dados {
    vector<float> imagens;
    vector<int> rotulos;

    size_t size() const { return rotulos.size(); }
    const float* imagem(size_t i) const { return &imagens[i * entradas]; }
};

Dados ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    Dados dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        array<float, entradas> imagem;
        int lidos = 0;
        int rotulo = -1;
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        for (; lidos < entradas; ++lidos) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            imagem[lidos] = valor / 16.0f;
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
        if (lidos != entradas ||
            !(campos >> rotulo) ||
            rotulo < 0 || rotulo >= classes ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.imagens.insert(dados.imagens.end(), imagem.begin(), imagem.end());
        dados.rotulos.push_back(rotulo);
    }
    if (arquivo.bad() || dados.size() == 0) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const float* escores, int rotulo, float* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            gradiente[i] = erro;
        }
    }
    return total;
}

// até 3 acertos e 3 erros por processo, cada um {linha, real, previsto}
constexpr int max_exemplos = 6;
constexpr int campos_exemplo = 3;

struct Parcial {
    int acertos = 0;
    double perda_total = 0;
    array<int, classes * classes> confusoes{};
    array<int, max_exemplos * campos_exemplo> exemplos{};
    int exemplos_acerto = 0;
    int exemplos_erro = 0;

    void processar(const float* escores, int rotulo, int indice_global) {
        // O índice do maior escore indica a classe prevista.
        const int previsto =
            static_cast<int>(max_element(escores, escores + classes) - escores);
        const bool acertou = previsto == rotulo;
        acertos += acertou;
        ++confusoes[rotulo * classes + previsto];
        perda_total += perda(escores, rotulo);

        int& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int* e = &exemplos[(exemplos_acerto + exemplos_erro) * campos_exemplo];
            e[0] = indice_global + 1;
            e[1] = rotulo;
            e[2] = previsto;
            ++exemplos_tipo;
        }
    }
};

// cada processo avalia uma fatia do teste
void avaliar(DenseLayer& l1, DenseLayer& l2, DenseLayer& l3,
             const Dados& teste, int rank, int total) {
    const auto inicio = chrono::steady_clock::now();

    // pesos do rank 0 para todos
    l1.compartilhar_parametros();
    l2.compartilhar_parametros();
    l3.compartilhar_parametros();

    int n = rank == 0 ? static_cast<int>(teste.size()) : 0;
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // fatias iguais; o que sobra fica com o rank 0
    const int bloco = n / total;
    const int distribuidos = bloco * total;
    Dados local;
    local.imagens.resize(static_cast<size_t>(bloco) * entradas);
    local.rotulos.resize(bloco);
    MPI_Scatter(teste.imagens.data(), bloco * entradas, MPI_FLOAT,
                local.imagens.data(), bloco * entradas, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    MPI_Scatter(teste.rotulos.data(), bloco, MPI_INT,
                local.rotulos.data(), bloco, MPI_INT,
                0, MPI_COMM_WORLD);

    Parcial parcial;
    for (int i = 0; i < bloco; ++i) {
        const float* escores = l3.Forward(l2.Forward(l1.Forward(local.imagem(i))));
        parcial.processar(escores, local.rotulos[i], rank * bloco + i);
    }
    if (rank == 0) {
        for (int i = distribuidos; i < n; ++i) {
            const float* escores = l3.Forward(l2.Forward(l1.Forward(teste.imagem(i))));
            parcial.processar(escores, teste.rotulos[i], i);
        }
    }

    int acertos = 0;
    double perda_total = 0;
    array<int, classes * classes> confusoes{};
    vector<int> todos_exemplos(rank == 0 ? total * max_exemplos * campos_exemplo : 0);
    MPI_Reduce(&parcial.acertos, &acertos, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&parcial.perda_total, &perda_total, 1, MPI_DOUBLE, MPI_SUM,
               0, MPI_COMM_WORLD);
    MPI_Reduce(parcial.confusoes.data(), confusoes.data(), classes * classes,
               MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Gather(parcial.exemplos.data(), max_exemplos * campos_exemplo, MPI_INT,
               todos_exemplos.data(), max_exemplos * campos_exemplo, MPI_INT,
               0, MPI_COMM_WORLD);
    const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

    if (rank != 0) {
        return;
    }

    // os 3 primeiros acertos e erros, na ordem das linhas
    struct Exemplo {
        int linha;
        int real;
        int previsto;
    };
    vector<Exemplo> candidatos;
    for (int k = 0; k < total * max_exemplos; ++k) {
        const int* e = &todos_exemplos[k * campos_exemplo];
        if (e[0] > 0) {
            candidatos.push_back({e[0], e[1], e[2]});
        }
    }
    sort(candidatos.begin(), candidatos.end(),
         [](const Exemplo& a, const Exemplo& b) { return a.linha < b.linha; });
    vector<Exemplo> exemplos;
    int exemplos_acerto = 0;
    int exemplos_erro = 0;
    for (const auto& c : candidatos) {
        int& exemplos_tipo = c.real == c.previsto ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            exemplos.push_back(c);
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / n
         << "% | " << acertos << " acertos em " << n << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / n << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        int ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            const int ocorrencias = confusoes[real * classes + previsto];
            if (real != previsto && ocorrencias > 0) {
                erros.push_back({real, previsto, ocorrencias});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
    cout << "Tempo de avaliacao " << setprecision(6) << tempo.count()
         << " s | " << total << " processo(s) MPI\n";
}

} // namespace Barenn

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, total;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total);

    if (argc != 1 && argc != 3) {
        if (rank == 0) {
            cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        }
        MPI_Finalize();
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    // só o rank 0 lê os arquivos; se falhar, todos encerram
    Barenn::Dados treino, teste;
    int falhou = 0;
    if (rank == 0) {
        try {
            treino = Barenn::ler_dados(arquivo_treino);
            teste = Barenn::ler_dados(arquivo_teste);
            if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
                throw runtime_error("Use arquivos distintos para treino e teste.");
            }
        } catch (const exception& erro) {
            cerr << "Erro: " << erro.what() << "\n";
            falhou = 1;
        }
    }
    MPI_Bcast(&falhou, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (falhou) {
        MPI_Finalize();
        return 1;
    }

    constexpr int epocas = 51;
    constexpr float taxa = 0.02f;
    // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
    Barenn::DenseLayer l1(Barenn::entradas, 32);
    Barenn::DenseLayer l2(32, 16);
    Barenn::DenseLayer l3(16, Barenn::classes, true);

    // o SGD é sequencial, então o treino fica só no rank 0
    if (rank == 0) {
        array<float, Barenn::classes> gradiente{};
        array<double, epocas> perdas{};

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (size_t i = 0; i < treino.size(); ++i) {
                const float* escores = l3.Forward(l2.Forward(l1.Forward(treino.imagem(i))));
                perda_total += Barenn::perda(escores, treino.rotulos[i], gradiente.data());

                // Propaga o gradiente da última camada até a primeira.
                // Cada Backward também atualiza os parâmetros da sua camada.
                const float* grad3 = l3.Backward(gradiente.data(), taxa);
                const float* grad2 = l2.Backward(grad3, taxa);
                l1.Backward(grad2, taxa, false);
            }
            perdas[epoca] = perda_total / treino.size();
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
    }

    Barenn::avaliar(l1, l2, l3, teste, rank, total);

    MPI_Finalize();
    return 0;
}
