import nltk
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from starlette.websockets import WebSocket

app = FastAPI()

dataset = [
    ("Hello", "Hi there! How can I assist you today?"),
    ("Goodbye", "Goodbye! Have a great day!"),
    ("What is your name?", "My name is Safari! I'm here to help you with your Tourism journey!"),
    ("What is Victoria Falls?", "Victoria Falls is a magnificent waterfall located on the Zambezi River in southern Africa. It is one of the largest and most famous waterfalls in the world."),
    ("Where is Victoria Falls located?", "Victoria Falls is located on the border of Zambia and Zimbabwe in southern Africa."),
    ("How tall is Victoria Falls?", "Victoria Falls has a height of approximately 108 meters (354 feet)."),
    ("What is the width of Victoria Falls?", "The width of Victoria Falls is about 1,708 meters (5,604 feet)."),
    ("What is the volume of water that flows through Victoria Falls?", "During the rainy season, Victoria Falls can have a peak flow rate of over 1 million liters (260,000 gallons) of water per second."),
    ("What is the best time to visit Victoria Falls?", "The best time to visit Victoria Falls is during the dry season, from May to October. The water levels are lower, providing better visibility and opportunities for various activities."),
    ("What is the rainy season at Victoria Falls?", "The rainy season at Victoria Falls typically occurs from November to April, with the highest rainfall in January and February."),
    ("What is the dry season at Victoria Falls?", "The dry season at Victoria Falls usually takes place from May to October, with minimal rainfall and lower water levels."),
    ("Is Victoria Falls a natural wonder?", "Yes, Victoria Falls is considered one of the Seven Natural Wonders of the World, known for its awe-inspiring beauty and grandeur."),
    ("Is Victoria Falls a UNESCO World Heritage Site?", "Yes, Victoria Falls is a UNESCO World Heritage Site, recognized for its exceptional natural beauty and geological significance."),
    ("What is another name for Victoria Falls?", "Victoria Falls is also known as Mosi-oa-Tunya, which means 'The Smoke That Thunders' in the local Tonga language."),
    ("How was Victoria Falls formed?", "Victoria Falls was formed through the process of erosion. The Zambezi River gradually carved out a deep gorge, resulting in the formation of the waterfall."),
    ("Can you swim in Victoria Falls?", "Swimming in Victoria Falls is not allowed due to the strong currents, turbulence, and the presence of wildlife, including crocodiles and hippos."),
    ("What activities can you do at Victoria Falls?", "There are various activities to enjoy at Victoria Falls, such as scenic helicopter flights, boat cruises, white-water rafting, bungee jumping, zip-lining, nature walks, and wildlife safaris."),
    ("Can you visit Victoria Falls from both Zambia and Zimbabwe?", "Yes, you can visit Victoria Falls from both the Zambian and Zimbabwean sides. Each side offers different viewpoints and experiences of the waterfall."),
    ("What are the main viewpoints of Victoria Falls?", "Some of the main viewpoints of Victoria Falls include the Knife-Edge Bridge, Devil's Pool, Livingstone Island, the Rainforest, and various scenic viewpoints along the falls."),
    ("Are there any accommodations near Victoria Falls?", "Yes, there are numerous accommodations available near Victoria Falls, ranging from luxury hotels and lodges to budget-friendly guesthouses and campsites."),
    ("What wildlife can be found near Victoria Falls?", "The area surrounding Victoria Falls is home to a diverse range of wildlife, including elephants, buffalos, zebras, giraffes, lions, leopards, hippos, crocodiles, and numerous bird species."),
    ("Is there an entrance fee to visit Victoria Falls?", "Yes, there is an entrance fee to visit Victoria Falls. The fee may vary depending on the nationality and the side (Zambia or Zimbabwe) you choose to visit."),
    ("Are guided tours available at Victoria Falls?", "Yes, there are various guided tours available at Victoria Falls, including guided walks, sunset cruises, cultural tours, and adventure activities."),
    ("What is the Victoria Falls Bridge?", "The Victoria Falls Bridge is a historic bridge that spans the Zambezi River, connecting Zambia and Zimbabwe. It offers breathtaking views of Victoria Falls and is also known for activities like bungee jumping and bridge tours."),
    ("Can you take a boat ride near Victoria Falls?", "Yes, there are boat rides available on the Zambezi River near Victoria Falls, providing an opportunity to experience the river's beauty, wildlife, and stunning sunsets."),
    ("What is the lunar rainbow at Victoria Falls?", "Under specific conditions during a full moon, you can witness a lunar rainbow, also known as a moonbow, at Victoria Falls. It is a beautiful phenomenon where the mist from the waterfall refracts the moonlight, creating a colorful rainbow at night."),
    ("What is the local currency at Victoria Falls?", "The local currencies used near Victoria Falls are the Zambian kwacha (ZMW) in Zambia and the Zimbabwean dollar (ZWL) in Zimbabwe."),
    ("What are some nearby attractions to visit along with Victoria Falls?", "Some nearby attractions to visit along with Victoria Falls include Chobe National Park in Botswana, Hwange National Park in Zimbabwe, and the town of Livingstone in Zambia."),
    ("Can you do a day trip to Victoria Falls from neighboring countries?", "Yes, it is possible to do day trips to Victoria Falls from neighboring countries, such as Botswana and Namibia. There are organized tours and transport options available."),
    ("What is the climate like at Victoria Falls?", "Victoria Falls has a subtropical climate. It can get hot during the summer months (October to April) with temperatures reaching around 30-40 degrees Celsius (86-104 degrees Fahrenheit), while the winter months (May to September) are cooler with temperatures ranging from 10-25 degrees Celsius (50-77 degrees Fahrenheit)."),
    ("What is the Zambezi River?", "The Zambezi River is the fourth-longest river in Africa, spanning six countries. It is the river on which Victoria Falls is located."),
    ("What is the local culture around Victoria Falls?", "The area around Victoria Falls is rich in culture. The local people have a vibrant heritage and traditions. You can explore the local customs, arts, crafts, music, and dance by visiting nearby villages and engaging with the communities."),
    ("Is Victoria Falls affected by climate change?", "Climate change has had some impact on Victoria Falls, affecting water levels and the timing of the rainy and dry seasons. However, the waterfall remains a remarkable natural wonder."),
    ("Are there any safety precautions to consider when visiting Victoria Falls?", "When visiting Victoria Falls, it's important to follow safety guidelines provided by authorities and tour operators. This includes staying on designated paths, avoiding swimming in restricted areas, and respecting wildlife."),
    ("What is the Victoria Falls Marathon?", "The Victoria Falls Marathon is an annual marathon race that takes place near Victoria Falls. It offers a scenic route with views of the falls and attracts participants from around the world."),
    ("Can you take a scenic flight over Victoria Falls?", "Yes, you can take a scenic helicopter or microlight flight over Victoria Falls for a breathtaking aerial view of the waterfall and the surrounding landscapes."),
    ("What is the Zambezi National Park?", "Zambezi National Park is a national park located on the Zambezi River, near Victoria Falls. It is home to a variety of wildlife, including elephants, buffalos, lions, and numerous bird species."),
("What are the popular tourist attractions in Zimbabwe?", "Some popular tourist attractions in Zimbabwe include Victoria Falls, Hwange National Park, Great Zimbabwe National Monument, Mana Pools National Park, and Matobo National Park."),
    ("When is the best time to visit Zimbabwe?", "The best time to visit Zimbabwe is during the dry season, from April to October, when wildlife is more concentrated around water sources. However, each season offers its own unique experiences."),
    ("What are the visa requirements for visiting Zimbabwe?", "Most visitors to Zimbabwe require a visa. You can obtain a visa on arrival at the airport or border post, or you can apply for an e-Visa in advance. Make sure to check the visa requirements based on your nationality."),
    ("Which currency is used in Zimbabwe?", "The official currency of Zimbabwe is the Zimbabwean dollar (ZWL). However, US dollars are widely accepted, especially in the tourism sector."),
    ("Are there any safety concerns for tourists in Zimbabwe?", "Zimbabwe is generally a safe country for tourists. However, it's advisable to take standard precautions such as being aware of your surroundings, avoiding isolated areas at night, and following any local guidance or restrictions."),
    ("What are the popular traditional dishes in Zimbabwe?", "Some popular traditional dishes in Zimbabwe include sadza (staple maize meal), nyama choma (grilled meat), madora (mopane worms), and bota (pumpkin porridge)."),
    ("Which airlines fly to Zimbabwe?", "Several international airlines fly to Zimbabwe, including South African Airways, Emirates, Ethiopian Airlines, Kenya Airways, and British Airways. There are also regional airlines and local carriers that operate within Zimbabwe."),
    ("What types of accommodations are available in Zimbabwe?", "Zimbabwe offers a range of accommodations, including luxury hotels, safari lodges, tented camps, guesthouses, and budget-friendly options. The choice depends on your preference and budget."),
    ("Can I go on a safari in Zimbabwe?", "Yes, Zimbabwe is renowned for its wildlife and offers excellent safari opportunities. You can go on game drives, walking safaris, or even canoe safaris in certain areas."),
    ("What are the popular adventure activities in Zimbabwe?", "Some popular adventure activities in Zimbabwe include white-water rafting on the Zambezi River, bungee jumping from Victoria Falls Bridge, zip-lining, canoeing, and hiking in national parks."),
    ("What is the climate like in Zimbabwe?",
     "Zimbabwe has a moderate climate, with warm summers and mild winters. The average temperature ranges from 25°C to 30°C (77°F to 86°F) during the summer months and 10°C to 20°C (50°F to 68°F) in winter."),
    ("What are the cultural festivals celebrated in Zimbabwe?",
     "Zimbabwe celebrates various cultural festivals throughout the year. Some popular festivals include the Harare International Festival of the Arts (HIFA), Victoria Falls Carnival, Shoko Festival, and Mbira Festival."),
    ("Are there any health concerns for travelers to Zimbabwe?",
     "It's recommended to check with your local health authorities or a travel health specialist for up-to-date information on vaccinations and health precautions. Malaria is prevalent in some areas, so taking appropriate measures is essential."),
    ("What is the local transportation like in Zimbabwe?",
     "Zimbabwe has a network of domestic flights, buses, and taxis to help you get around. In cities like Harare and Bulawayo, there are public transportation systems, including minibuses and taxis. Private car hire services are also available."),
    ("What are the popular arts and crafts of Zimbabwe?",
     "Zimbabwean arts and crafts are diverse and include stone sculptures, traditional pottery, basket weaving, and intricate wood carvings. The Shona sculpture is particularly renowned worldwide."),
    ("Can I visit local communities or interact with tribes in Zimbabwe?",
     "Yes, you can visit local communities and interact with tribes in Zimbabwe. There are cultural tourism initiatives that offer opportunities to experience traditional customs, dances, and lifestyles of different ethnic groups."),
    ("What is the electricity voltage in Zimbabwe?",
     "The electricity voltage in Zimbabwe is 220-240 volts, with a frequency of 50 Hz. The power plugs and sockets used are of the type D and G."),
    ("Are there any restrictions on photography in national parks?",
     "Photography is generally allowed in national parks and game reserves in Zimbabwe. However, some parks may have specific guidelines and restrictions, especially when photographing certain wildlife species or during guided walks."),
    ("Can I go fishing in Zimbabwe?",
     "Yes, fishing is a popular activity in Zimbabwe, particularly in Lake Kariba and the Zambezi River. You can catch species such as tigerfish, bream, and catfish."),
    ("What is the local time zone in Zimbabwe?",
     "Zimbabwe operates on Central Africa Time (CAT), which is UTC+2 throughout the year."),
    ("What is the distance between Harare and Victoria Falls?",
     "The distance between Harare and Victoria Falls is approximately 900 kilometers (560 miles) by road."),
    ("Are there any luxury train services in Zimbabwe?",
     "Yes, Zimbabwe offers luxury train services such as the Rovos Rail and the Zimbabwe Railways' Victoria Falls Steam Train. These provide a unique and elegant way to experience the scenic beauty of the country."),
    ("What is the local cuisine like in Zimbabwe?",
     "The local cuisine in Zimbabwe is diverse and influenced by traditional dishes. Some popular dishes include sadza, roasted meats, peanut butter stew (dovi), and local delicacies like fried kapenta fish."),
    ("Which national parks offer the best wildlife viewing in Zimbabwe?",
     "Hwange National Park, Mana Pools National Park, and Gonarezhou National Park are known for their exceptional wildlife viewing opportunities, offering a chance to spot elephants, lions, buffalo, giraffes, and other animals."),
    ("Are there any wine estates or vineyards in Zimbabwe?",
     "Yes, there are a few wine estates and vineyards in Zimbabwe, particularly in the Eastern Highlands region. These produce a range of wines, including Chenin Blanc, Chardonnay, and Merlot."),
    ("What is the official language spoken in Zimbabwe?",
     "The official language of Zimbabwe is English. However, there are also several indigenous languages spoken, including Shona and Ndebele."),
    ("Can I visit the Great Zimbabwe ruins?",
     "Yes, you can visit the Great Zimbabwe National Monument, a UNESCO World Heritage Site. It is an ancient city built of stone, known for its impressive architecture and historical significance."),
    ("Are there any golf courses in Zimbabwe?",
     "Zimbabwe offers a number of golf courses, including championship courses. Some popular ones include the Royal Harare Golf Club, Borrowdale Brooke Golf Club, and Leopard Rock Golf Course."),
    ("What adventure activities are available at Victoria Falls?",
     "At Victoria Falls, you can enjoy thrilling activities such as white-water rafting, zip-lining, helicopter tours, micro-light flights, and sunset cruises on the Zambezi River."),
    ("Can I go on a hot air balloon safari in Zimbabwe?",
     "Yes, hot air balloon safaris are available in some parts of Zimbabwe, offering a unique and scenic perspective of the wildlife and landscapes."),
    ("What are the major cities in Zimbabwe?",
     "The major cities in Zimbabwe include Harare (the capital), Bulawayo, Mutare, Gweru, Masvingo, and Victoria Falls."),
    ("Can I climb Mount Nyangani?",
     "Yes, Mount Nyangani is the highest peak in Zimbabwe, and climbing it is a popular activity for outdoor enthusiasts."),
    ("What is the currency exchange rate in Zimbabwe?",
     "The currency exchange rate in Zimbabwe can vary, so it's recommended to check with authorized foreign exchange dealers or banks for the latest rates."),
    ("What is the dress code for visiting national parks in Zimbabwe?",
     "It's advisable to wear comfortable and lightweight clothing, preferably in neutral colors, when visiting national parks. Also, don't forget to bring sturdy footwear and a hat for sun protection."),
    ("Are there any cultural museums in Zimbabwe?",
     "Yes, there are several cultural museums in Zimbabwe that showcase the rich history, art, and traditions of the country. Some notable ones include the National Museum and the Zimbabwe Museum of Human Sciences."),
    ("Can I go on a walking safari in Zimbabwe?",
     "Walking safaris are available in certain areas of Zimbabwe, offering an up-close and immersive wildlife experience. It's an opportunity to explore the bush on foot with a professional guide."),
    ("Are there any scenic drives in Zimbabwe?",
     "Zimbabwe offers breathtaking scenic drives, such as the Eastern Highlands route, the road to Kariba Dam, and the drive through Matobo Hills. These routes provide stunning views of mountains, lakes, and unique landscapes."),
    ("What is the average rainfall in Zimbabwe?",
     "The average annual rainfall in Zimbabwe ranges from 600mm to 1,000mm (23.6 to 39.4 inches), with regional variations."),
    ("Can I go on a horseback safari in Zimbabwe?",
     "Yes, horseback safaris are available in certain areas of Zimbabwe, providing a unique way to explore the wilderness and spot wildlife while riding through the bush."),
    ("What is the local etiquette and customs in Zimbabwe?",
     "It's respectful to greet people with a handshake and to use appropriate titles when addressing others. It's also customary to ask for permission before taking photos of individuals or their property."),
    ("Can I visit traditional craft markets in Zimbabwe?",
     "Yes, there are traditional craft markets where you can find a variety of handmade crafts, including wood carvings, beadwork, woven baskets, and traditional clothing."),
    ("Are there any art galleries in Zimbabwe?",
     "Zimbabwe has a vibrant art scene, and there are several art galleries in major cities like Harare and Bulawayo that showcase the works of local artists."),
    ("Can I go on a canoe safari on the Zambezi River?",
     "Yes, canoe safaris are available on the Zambezi River, offering a peaceful and unique way to observe wildlife and enjoy the scenery."),
    ("What is the best time of year to visit Zimbabwe?",
        "The best time to visit Zimbabwe is during the dry season, from May to October. This is when wildlife viewing is at its best, and the weather is pleasant."),
("What is the distance between Harare and Victoria Falls?", "The distance between Harare and Victoria Falls is approximately 900 kilometers (560 miles) by road."),
    ("Are there any luxury train services in Zimbabwe?", "Yes, Zimbabwe offers luxury train services such as the Rovos Rail and the Zimbabwe Railways' Victoria Falls Steam Train. These provide a unique and elegant way to experience the scenic beauty of the country."),
    ("What is the local cuisine like in Zimbabwe?", "The local cuisine in Zimbabwe is diverse and influenced by traditional dishes. Some popular dishes include sadza, roasted meats, peanut butter stew (dovi), and local delicacies like fried kapenta fish."),
    ("Which national parks offer the best wildlife viewing in Zimbabwe?", "Hwange National Park, Mana Pools National Park, and Gonarezhou National Park are known for their exceptional wildlife viewing opportunities, offering a chance to spot elephants, lions, buffalo, giraffes, and other animals."),
    ("Are there any wine estates or vineyards in Zimbabwe?", "Yes, there are a few wine estates and vineyards in Zimbabwe, particularly in the Eastern Highlands region. These produce a range of wines, including Chenin Blanc, Chardonnay, and Merlot."),
    ("What is the official language spoken in Zimbabwe?", "The official language of Zimbabwe is English. However, there are also several indigenous languages spoken, including Shona and Ndebele."),
    ("Can I visit the Great Zimbabwe ruins?", "Yes, you can visit the Great Zimbabwe National Monument, a UNESCO World Heritage Site. It is an ancient city built of stone, known for its impressive architecture and historical significance."),
    ("Are there any golf courses in Zimbabwe?", "Zimbabwe offers a number of golf courses, including championship courses. Some popular ones include the Royal Harare Golf Club, Borrowdale Brooke Golf Club, and Leopard Rock Golf Course."),
    ("What adventure activities are available at Victoria Falls?", "At Victoria Falls, you can enjoy thrilling activities such as white-water rafting, zip-lining, helicopter tours, micro-light flights, and sunset cruises on the Zambezi River."),
    ("Can I go on a hot air balloon safari in Zimbabwe?", "Yes, hot air balloon safaris are available in some parts of Zimbabwe, offering a unique and scenic perspective of the wildlife and landscapes."),
    ("What are the major cities in Zimbabwe?", "The major cities in Zimbabwe include Harare (the capital), Bulawayo, Mutare, Gweru, Masvingo, and Victoria Falls."),
    ("Can I climb Mount Nyangani?", "Yes, Mount Nyangani is the highest peak in Zimbabwe, and climbing it is a popular activity for outdoor enthusiasts."),
    ("What is the currency exchange rate in Zimbabwe?", "The currency exchange rate in Zimbabwe can vary, so it's recommended to check with authorized foreign exchange dealers or banks for the latest rates."),
    ("What is the dress code for visiting national parks in Zimbabwe?", "It's advisable to wear comfortable and lightweight clothing, preferably in neutral colors, when visiting national parks. Also, don't forget to bring sturdy footwear and a hat for sun protection."),
    ("Are there any cultural museums in Zimbabwe?", "Yes, there are several cultural museums in Zimbabwe that showcase the rich history, art, and traditions of the country. Some notable ones include the National Museum and the Zimbabwe Museum of Human Sciences."),
    ("Can I go on a walking safari in Zimbabwe?", "Walking safaris are available in certain areas of Zimbabwe, offering an up-close and immersive wildlife experience. It's an opportunity to explore the bush on foot with a professional guide."),
    ("Are there any scenic drives in Zimbabwe?", "Zimbabwe offers breathtaking scenic drives, such as the Eastern Highlands route, the road to Kariba Dam, and the drive through Matobo Hills. These routes provide stunning views of mountains, lakes, and unique landscapes."),
    ("What is the average rainfall in Zimbabwe?", "The average annual rainfall in Zimbabwe ranges from 600mm to 1,000mm (23.6 to 39.4 inches), with regional variations."),
    ("Can I go on a horseback safari in Zimbabwe?", "Yes, horseback safaris are available in certain areas of Zimbabwe, providing a unique way to explore the wilderness and spot wildlife while riding through the bush."),
    ("What is the local etiquette and customs in Zimbabwe?", "It's respectful to greet people with a handshake and to use appropriate titles when addressing others. It's also customary to ask for permission before taking photos of individuals or their property."),
    ("Can I visit traditional craft markets in Zimbabwe?", "Yes, there are traditional craft markets where you can find a variety of handmade crafts, including wood carvings, beadwork, woven baskets, and traditional clothing."),
    ("Are there any art galleries in Zimbabwe?", "Zimbabwe has a vibrant art scene, and there are several art galleries in major cities like Harare and Bulawayo that showcase the works of local artists."),
    ("Can I go on a canoe safari on the Zambezi River?", "Yes, canoe safaris are available on the Zambezi River, offering a peaceful and unique way to observe wildlife, bird species, and the beautiful river ecosystem."),
    ("What is the local time zone in Zimbabwe?", "Zimbabwe operates on Central Africa Time (CAT), which is two hours ahead of Greenwich Mean Time (GMT+2)."),
    ("What is the distance between Harare and Victoria Falls?",
     "The distance between Harare and Victoria Falls is approximately 900 kilometers (560 miles) by road."),
    ("Are there any luxury train services in Zimbabwe?",
     "Yes, Zimbabwe offers luxury train services such as the Rovos Rail and the Zimbabwe Railways' Victoria Falls Steam Train. These provide a unique and elegant way to experience the scenic beauty of the country."),
    ("What is the local cuisine like in Zimbabwe?",
     "The local cuisine in Zimbabwe is diverse and influenced by traditional dishes. Some popular dishes include sadza, roasted meats, peanut butter stew (dovi), and local delicacies like fried kapenta fish."),
    ("Which national parks offer the best wildlife viewing in Zimbabwe?",
     "Hwange National Park, Mana Pools National Park, and Gonarezhou National Park are known for their exceptional wildlife viewing opportunities, offering a chance to spot elephants, lions, buffalo, giraffes, and other animals."),
    ("Are there any wine estates or vineyards in Zimbabwe?",
     "Yes, there are a few wine estates and vineyards in Zimbabwe, particularly in the Eastern Highlands region. These produce a range of wines, including Chenin Blanc, Chardonnay, and Merlot."),
    ("What is the official language spoken in Zimbabwe?",
     "The official language of Zimbabwe is English. However, there are also several indigenous languages spoken, including Shona and Ndebele."),
    ("Can I visit the Great Zimbabwe ruins?",
     "Yes, you can visit the Great Zimbabwe National Monument, a UNESCO World Heritage Site. It is an ancient city built of stone, known for its impressive architecture and historical significance."),
    ("Are there any golf courses in Zimbabwe?",
     "Zimbabwe offers a number of golf courses, including championship courses. Some popular ones include the Royal Harare Golf Club, Borrowdale Brooke Golf Club, and Leopard Rock Golf Course."),
    ("What adventure activities are available at Victoria Falls?",
     "At Victoria Falls, you can enjoy thrilling activities such as white-water rafting, zip-lining, helicopter tours, micro-light flights, and sunset cruises on the Zambezi River."),
    ("Can I go on a hot air balloon safari in Zimbabwe?",
     "Yes, hot air balloon safaris are available in some parts of Zimbabwe, offering a unique and scenic perspective of the wildlife and landscapes."),
    ("What are the major cities in Zimbabwe?",
     "The major cities in Zimbabwe include Harare (the capital), Bulawayo, Mutare, Gweru, Masvingo, and Victoria Falls."),
    ("Can I climb Mount Nyangani?",
     "Yes, Mount Nyangani is the highest peak in Zimbabwe, and climbing it is a popular activity for outdoor enthusiasts."),
    ("What is the currency exchange rate in Zimbabwe?",
     "The currency exchange rate in Zimbabwe can vary, so it's recommended to check with authorized foreign exchange dealers or banks for the latest rates."),
    ("What is the dress code for visiting national parks in Zimbabwe?",
     "It's advisable to wear comfortable and lightweight clothing, preferably in neutral colors, when visiting national parks. Also, don't forget to bring sturdy footwear and a hat for sun protection."),
    ("Are there any cultural museums in Zimbabwe?",
     "Yes, there are several cultural museums in Zimbabwe that showcase the rich history, art, and traditions of the country. Some notable ones include the National Museum and the Zimbabwe Museum of Human Sciences."),
    ("Can I go on a walking safari in Zimbabwe?",
     "Walking safaris are available in certain areas of Zimbabwe, offering an up-close and immersive wildlife experience. It's an opportunity to explore the bush on foot with a professional guide."),
("Are there any scenic drives in Zimbabwe?", "Zimbabwe offers breathtaking scenic drives, such as the Eastern Highlands route, the road to Kariba Dam, and the drive through Matobo Hills. These routes provide stunning views of mountains, lakes, and unique landscapes."),
    ("What is the average rainfall in Zimbabwe?", "The average annual rainfall in Zimbabwe ranges from 600mm to 1,000mm (23.6 to 39.4 inches), with regional variations."),
    ("Can I go on a horseback safari in Zimbabwe?", "Yes, horseback safaris are available in certain areas of Zimbabwe, providing a unique way to explore the wilderness and spot wildlife while riding through the bush."),
    ("What is the local etiquette and customs in Zimbabwe?", "It's respectful to greet people with a handshake and to use appropriate titles when addressing others. It's also customary to ask for permission before taking photos of individuals or their property."),
    ("Can I visit traditional craft markets in Zimbabwe?", "Yes, there are traditional craft markets where you can find a variety of handmade crafts, including wood carvings, beadwork, woven baskets, and traditional clothing."),
    ("Are there any art galleries in Zimbabwe?", "Zimbabwe has a vibrant art scene, and there are several art galleries in major cities like Harare and Bulawayo that showcase the works of local artists."),
    ("Can I go on a canoe safari on the Zambezi River?", "Yes, canoe safaris are available on the Zambezi River, offering a peaceful and unique way to observe wildlife, bird species, and the beautiful river ecosystem."),
("What is the local time zone in Zimbabwe?", "Zimbabwe operates on Central Africa Time (CAT), which is UTC+2 throughout the year."),
    ("Can I visit the Chinhoyi Caves?", "Yes, the Chinhoyi Caves are open to visitors. These limestone caves are known for their stunning underground lake and are a popular spot for diving and exploration."),
    ("Are there any traditional ceremonies or festivals in Zimbabwe?", "Yes, Zimbabwe has a rich cultural heritage, and various traditional ceremonies and festivals are celebrated throughout the year. Some notable ones include the Kwanza ceremony of the Shona people and the Umhlanga (Reed Dance) of the Ndebele people."),
    ("What is the best way to travel between cities in Zimbabwe?", "Traveling by road is common for shorter distances, while domestic flights are available for longer distances. Bus services and private taxis are also options for intercity travel."),
    ("Can I visit a local village and interact with the community?", "Yes, there are opportunities to visit local villages and interact with the community, especially in rural areas. It's a chance to learn about their way of life, traditions, and customs."),
    ("What is the tipping culture in Zimbabwe?", "Tipping is not mandatory in Zimbabwe, but it's appreciated for good service. It's common to tip around 10% of the bill at restaurants and to tip guides and drivers on safari."),
    ("Can I go on a fishing safari in Zimbabwe?", "Yes, fishing safaris are available in Zimbabwe, particularly in Lake Kariba and the Zambezi River, where you can catch species like tigerfish and bream."),

]

nltk.download("punkt")
sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")


def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return tokens

corpus = [preprocess(pair[0]) for pair in dataset]
vectorizer = TfidfVectorizer(tokenizer=preprocess)
X = vectorizer.fit_transform([" ".join(tokens) for tokens in corpus])


def generate_response(user_query):
    user_tokens = preprocess(user_query)
    user_query_vector = vectorizer.transform([" ".join(user_tokens)])

    similarities = cosine_similarity(user_query_vector, X).flatten()
    best_match_index = similarities.argmax()

    response = dataset[best_match_index][1]
    return response

@app.websocket("/api/chatbot")
async def chatbot_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        try:
            message = await websocket.receive_text()
            response = generate_response(message)
            await websocket.send_text(response)
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    await websocket.close()
