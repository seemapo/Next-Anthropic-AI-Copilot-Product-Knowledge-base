import { Pinecone } from '@pinecone-database/pinecone';
import {posts} from "@/app/lib/data/data";
import { OpenAIAdapter } from "@copilotkit/runtime"; // ✅ use Azure-specific adapter
import { copilotRuntimeNextJSAppRouterEndpoint } from "@copilotkit/runtime";
import { CopilotRuntime } from "@copilotkit/runtime";
import { NextRequest } from "next/server";
import OpenAI from 'openai';

const PINECONE_API_KEY = process.env.NEXT_PUBLIC_PINECONE_API_KEY;

const AZURE_OPENAI_ENDPOINT = process.env.AZURE_OPENAI_ENDPOINT!;
const AZURE_OPENAI_DEPLOYMENT_NAME = process.env.AZURE_OPENAI_DEPLOYMENT_NAME!;
const AZURE_OPENAI_API_VERSION = "2024-02-15-preview"; 
const OPENAI_API_KEY = process.env.NEXT_PUBLIC_AZURE_OPENAI_API_KEY;
const AZURE_OPENAI_BASEURL= `${AZURE_OPENAI_ENDPOINT}openai/deployments/${AZURE_OPENAI_DEPLOYMENT_NAME}`;

console.log('AZURE_OPENAI_ENDPOINT:', AZURE_OPENAI_ENDPOINT);
console.log('AZURE_OPENAI_DEPLOYMENT_NAME:', AZURE_OPENAI_DEPLOYMENT_NAME);
console.log('AZURE_OPENAI_API_VERSION:', AZURE_OPENAI_API_VERSION);
console.log('OPENAI_API_KEY:', OPENAI_API_KEY);
console.log('AZURE_OPENAI_BASEURL:', AZURE_OPENAI_BASEURL);


const openai = new OpenAI({
    apiKey: OPENAI_API_KEY,
    baseURL: AZURE_OPENAI_BASEURL,
    defaultQuery: { "api-version": "2024-04-01-preview" },
    defaultHeaders: { "api-key": OPENAI_API_KEY },
  });
 
  const serviceAdapter = new OpenAIAdapter({
    openai    });

 
if (!OPENAI_API_KEY || !PINECONE_API_KEY) {
    console.error('Missing required API keys. ');
    process.exit(1);
  }

const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
const model = 'multilingual-e5-large';
const indexName = 'knowledge-base-data';

const initializePinecone = async () => {
    const maxRetries = 3;
    const retryDelay = 2000;
  
    for (let i = 0; i < maxRetries; i++) {
      try {
        const indexList = await pinecone.listIndexes();
        if (!indexList.indexes?.some(index => index.name === indexName)) {
          await pinecone.createIndex({
            name: indexName,
            dimension: 1024,
            metric: 'cosine',
            spec: {
              serverless: {
                cloud: 'aws',
                region: 'us-east-1',
              },
            },
          });
          await new Promise(resolve => setTimeout(resolve, 5000));
        }
        return pinecone.index(indexName);
      } catch (error) {
        if (i === maxRetries - 1) throw error;
        console.warn(`Retrying Pinecone initialization... (${i + 1}/${maxRetries})`);
        await new Promise(resolve => setTimeout(resolve, retryDelay));
      }
    }
    return null; 
  };

  // Initialize Pinecone and prepare the index
(async () => {
    try {
      const index = await initializePinecone();
      if (index) {
        const embeddingsResponse =  await pinecone.inference.embed(
          model,
          posts.map(d => d.content),
          { inputType: 'passage', truncate: 'END' }
        ) ;

        const embeddings = embeddingsResponse.data;
        const records = posts.map((d, i) => {
            const embedding = embeddings[i];
            return {
            id: d.id.toString(),
            values:  embedding && embedding.values ? embedding.values : [],
            metadata: { text: d.content },
            };
    });  
        await index.namespace('knowledge-base-data-namespace').upsert(
          records.map(record => ({
            ...record,
            values: record.values || [],
          }))
        );
      }
    } catch (error) {
      console.error('Error initializing Pinecone:', error);
      process.exit(1);
    }
  })();

  const runtime = new CopilotRuntime({
    actions: () => [
      {
        name: 'FetchKnowledgebaseArticles',
        description: 'Fetch relevant knowledge base articles based on a user query',
        parameters: [
          {
            name: 'query',
            type: 'string',
            description: 'The User query for the knowledge base index search to perform',
            required: true,
          },
        ],
        handler: async ({ query }: { query: string }) => {
          try {
            const queryEmbeddingResponse = await pinecone.inference.embed(
              model,
              [query],
              { inputType: 'query' }
            );

            const queryEmbedding = queryEmbeddingResponse.data[0].values;
            if (!Array.isArray(queryEmbedding) || queryEmbedding.length === 0) {
                throw new Error('Invalid embedding: Expected a non-empty array of numbers.');
              }
            console.log('queryEmbedding:', queryEmbedding);
            const queryResponse = await pinecone
              .index(indexName)
              .namespace('knowledge-base-data-namespace')
              .query({
                topK: 3,
                vector: queryEmbedding || [],
                includeValues: false,
                includeMetadata: true,
              });
            return { articles: queryResponse?.matches || [] };
          } catch (error) {
            console.error('Error fetching knowledge base articles:', error);
            throw new Error('Failed to fetch knowledge base articles.');
          }
        },    },
    ],
  });
  

  export const POST = async (req: NextRequest) => {
    const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
      runtime,
      serviceAdapter,
      endpoint: '/api/copilotkit',
    });
    
    console.log("Request received:", req);
    return handleRequest(req);
       
    
  };