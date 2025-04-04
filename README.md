# CA-GCN

flowchart TD
    %% Data Layer Subgraph
    subgraph "Data Layer (KG Datasets)"
        D1["data/ICEWS14"]
        D2["data/ICEWS18"]
        D3["data/WIKI"]
        D4["data/YAGO"]
        D5["data/GDELT"]
    end

    %% Preprocessing & History Extraction Subgraph
    subgraph "Preprocessing & History Extraction"
        P1["get_history_record.py"]
        P2["data/ICEWS14/history_seq"]
    end

    %% GCN Model Engine Subgraph
    subgraph "GCN Model Engine"
        M1["rgcn/knowledge_graph.py"]
        M2["rgcn/layers.py"]
        M3["rgcn/model.py"]
        M4["rgcn/utils.py"]
        M5["src/model.py"]
        M6["src/rrgcn.py"]
    end

    %% Training / Inference Pipeline
    T1["src/main.py"]

    %% Output and Logging Subgraph
    subgraph "Output and Logging"
        O1["models/"]
        O2["logs/"]
    end

    %% Connections
    D1 --> P1
    D2 --> P1
    D3 --> P1
    D4 --> P1
    D5 --> P1

    P1 --> T1
    P2 --> T1

    T1 -->|"uses"| M1
    T1 -->|"uses"| M2
    T1 -->|"uses"| M3
    T1 -->|"uses"| M4
    T1 -->|"uses"| M5
    T1 -->|"uses"| M6

    T1 --> O1
    T1 --> O2

    %% Styles
    classDef dataLayer fill:#f9c74f,stroke:#000,stroke-width:2px;
    classDef preprocessing fill:#90be6d,stroke:#000,stroke-width:2px;
    classDef modelEngine fill:#577590,stroke:#000,stroke-width:2px;
    classDef training fill:#f94144,stroke:#000,stroke-width:2px;
    classDef output fill:#277da1,stroke:#000,stroke-width:2px;

    class D1,D2,D3,D4,D5 dataLayer;
    class P1,P2 preprocessing;
    class M1,M2,M3,M4,M5,M6 modelEngine;
    class T1 training;
    class O1,O2 output;

    %% Click Events
    click D1 "https://github.com/echkidieu1102/ca-gcn/tree/main/data/ICEWS14"
    click D2 "https://github.com/echkidieu1102/ca-gcn/tree/main/data/ICEWS18"
    click D3 "https://github.com/echkidieu1102/ca-gcn/tree/main/data/WIKI"
    click D4 "https://github.com/echkidieu1102/ca-gcn/tree/main/data/YAGO"
    click D5 "https://github.com/echkidieu1102/ca-gcn/tree/main/data/GDELT"
    click P1 "https://github.com/echkidieu1102/ca-gcn/blob/main/get_history_record.py"
    click P2 "https://github.com/echkidieu1102/ca-gcn/tree/main/data/ICEWS14/history_seq"
    click M1 "https://github.com/echkidieu1102/ca-gcn/blob/main/rgcn/knowledge_graph.py"
    click M2 "https://github.com/echkidieu1102/ca-gcn/blob/main/rgcn/layers.py"
    click M3 "https://github.com/echkidieu1102/ca-gcn/blob/main/rgcn/model.py"
    click M4 "https://github.com/echkidieu1102/ca-gcn/blob/main/rgcn/utils.py"
    click M5 "https://github.com/echkidieu1102/ca-gcn/blob/main/src/model.py"
    click M6 "https://github.com/echkidieu1102/ca-gcn/blob/main/src/rrgcn.py"
    click T1 "https://github.com/echkidieu1102/ca-gcn/blob/main/src/main.py"
    click O1 "https://github.com/echkidieu1102/ca-gcn/tree/main/models/"
    click O2 "https://github.com/echkidieu1102/ca-gcn/tree/main/logs/"
