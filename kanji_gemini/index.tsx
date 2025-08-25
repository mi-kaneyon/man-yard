import React, { useState, useRef, FormEvent, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import { GoogleGenAI } from "@google/genai";

const App: React.FC = () => {
    type GameState = 'setup' | 'loading' | 'playing' | 'feedback' | 'level-transition' | 'error';
    type QuestionType = 'onyomi' | 'kunyomi' | 'writing';
    type Scene = {
        story: string;
        imagePrompt: string;
        kanji: string;
        questionType: QuestionType;
        correctAnswer: string;
        questionWord: string;
    };

    const [gameState, setGameState] = useState<GameState>('setup');
    const [gradeLevel, setGradeLevel] = useState<number>(1);
    const [level, setLevel] = useState<number>(1);
    const [questionNumber, setQuestionNumber] = useState<number>(1);
    const [correctAnswers, setCorrectAnswers] = useState<number>(0);
    const [maxGradeUnlocked, setMaxGradeUnlocked] = useState<number>(1);
    
    const [currentScene, setCurrentScene] = useState<Scene | null>(null);
    const [currentImage, setCurrentImage] = useState<string | null>(null);
    const [userAnswer, setUserAnswer] = useState<string>('');
    const [feedback, setFeedback] = useState<'correct' | 'incorrect' | null>(null);
    const [error, setError] = useState<string | null>(null);

    const kanjiHistory = useRef<string[]>([]);
    const storyHistory = useRef<string[]>([]);

    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

    const generateNextScene = async (grade: number) => {
        setGameState('loading');
        setUserAnswer('');
        setFeedback(null);
        setCurrentImage(null);
        
        try {
            const systemInstruction = `あなたは「漢字クエスト」のゲームマスターです。日本の小学生向けの漢字学習ゲームをファンタジーの世界観で進行します。Google検索を使い、正確な情報を元に問題を作成してください。`;

            const userPrompt = `
                日本の小学${grade}年生が習う漢字をGoogle検索で調べてください。
                ただし、このリストにある漢字は使わないでください: [${kanjiHistory.current.join(', ') || 'なし'}]
                
                検索結果から適切な漢字を1つ選び、["onyomi", "kunyomi", "writing"] からランダムに問題形式を1つ選択してください。
                
                選んだ形式に応じて、以下の要件でファンタジーの物語のワンシーンと問題を作成してください。
                1. 短く（2〜3文）エキサイティングな物語。
                2. 物語の情景を表す、英語の画像生成プロンプト（スタイル: "epic fantasy, vibrant, digital art, cinematic lighting"）。
                3. 「onyomi」「kunyomi」の場合：物語に漢字を使い、答えは「ひらがな」の読み。
                4. 「writing」の場合：物語に単語の「ひらがな」を使い、答えは「漢字」。
                5. 問題で使う単語（漢字またはひらがな）。

                結果は、以下の形式で「|||」を区切り文字として厳密に出力してください。他のテキストは含めないでください。
                STORY:[物語]|||IMAGE_PROMPT:[画像プロンプト]|||KANJI:[漢字]|||QUESTION_TYPE:[onyomi|kunyomi|writing]|||CORRECT_ANSWER:[答え]|||QUESTION_WORD:[問題で使う単語]
            `;
            
            const response = await ai.models.generateContent({
                model: "gemini-2.5-flash",
                contents: userPrompt,
                config: {
                    systemInstruction,
                    tools: [{ googleSearch: {} }],
                }
            });

            const text = response.text.trim();
            const parts = text.split('|||');
            const sceneData: Partial<Scene> = {};
            
            parts.forEach(part => {
                const [key, ...valueParts] = part.split(':');
                const value = valueParts.join(':').trim();
                if (key && value) {
                    switch (key.trim()) {
                        case 'STORY': sceneData.story = value; break;
                        case 'IMAGE_PROMPT': sceneData.imagePrompt = value; break;
                        case 'KANJI': sceneData.kanji = value; break;
                        case 'QUESTION_TYPE':
                            if (['onyomi', 'kunyomi', 'writing'].includes(value)) {
                                sceneData.questionType = value as QuestionType;
                            }
                            break;
                        case 'CORRECT_ANSWER': sceneData.correctAnswer = value; break;
                        case 'QUESTION_WORD': sceneData.questionWord = value; break;
                    }
                }
            });

            if (!sceneData.story || !sceneData.imagePrompt || !sceneData.kanji || !sceneData.questionType || !sceneData.correctAnswer || !sceneData.questionWord) {
                throw new Error(`Failed to parse scene data from AI response. Response was: "${text}"`);
            }

            setCurrentScene(sceneData as Scene);
            generateImage(sceneData.imagePrompt);
            storyHistory.current.push(sceneData.story);
            kanjiHistory.current.push(sceneData.kanji);
            
            setGameState('playing');

        } catch (err) {
            console.error("Error generating scene:", err);
            setError("物語の生成に失敗しました。ページを再読み込みしてください。");
            setGameState('error');
        }
    };
    
    const generateImage = async (prompt: string) => {
        try {
            const imageResponse = await ai.models.generateImages({
                model: 'imagen-3.0-generate-002',
                prompt: prompt,
                config: {
                    numberOfImages: 1,
                    outputMimeType: 'image/jpeg',
                    aspectRatio: '16:9',
                },
            });
            const base64ImageBytes: string = imageResponse.generatedImages[0].image.imageBytes;
            const imageUrl = `data:image/jpeg;base64,${base64ImageBytes}`;
            setCurrentImage(imageUrl);
        } catch (err) {
            console.error("Error generating image:", err);
            // Don't block the game if image fails.
            setCurrentImage(null); 
        }
    }

    const startLevel = (grade: number, newLevel: number) => {
        setGradeLevel(grade);
        setLevel(newLevel);
        setQuestionNumber(1);
        setCorrectAnswers(0);
        kanjiHistory.current = [];
        storyHistory.current = [];
        generateNextScene(grade);
    }
    
    const handleSubmit = (e: FormEvent) => {
        e.preventDefault();
        if (!currentScene || feedback) return;

        const isCorrect = userAnswer.trim() === currentScene.correctAnswer;
        if (isCorrect) {
            setCorrectAnswers(prev => prev + 1);
        }
        setFeedback(isCorrect ? 'correct' : 'incorrect');

        setTimeout(() => {
            const nextQuestionNum = questionNumber + 1;
            if (nextQuestionNum > 20) {
                setGameState('level-transition');
            } else {
                setQuestionNumber(nextQuestionNum);
                generateNextScene(gradeLevel);
            }
        }, 1500);
    }

    const renderQuestion = () => {
        if (!currentScene) return "";
        switch (currentScene.questionType) {
            case 'onyomi': return `この漢字の「音読み」は何ですか？`;
            case 'kunyomi': return `この漢字の「訓読み」は何ですか？`;
            case 'writing': return `「${currentScene.questionWord}」を漢字で書くと？`;
            default: return "";
        }
    }
    
    const renderStoryWithHighlight = (story: string, word: string) => {
        if (!story || !word) return "";
        const parts = story.split(word);
        return parts.map((part, index) => (
            <React.Fragment key={index}>
                {part}
                {index < parts.length - 1 && <span className="kanji-highlight">{word}</span>}
            </React.Fragment>
        ));
    }
    
    const renderLevelTransition = () => {
        const isWin = correctAnswers >= 18; // 90% of 20
        
        if (isWin) {
            const isLastLevel = level === 5;
            const isLastGrade = gradeLevel === 6;

            const handleNext = () => {
                if (isLastLevel) {
                    if (!isLastGrade) {
                        const nextGrade = gradeLevel + 1;
                        setMaxGradeUnlocked(prev => Math.max(prev, nextGrade));
                        startLevel(nextGrade, 1);
                    } else {
                        // Game complete!
                        setGameState('setup');
                    }
                } else {
                    startLevel(gradeLevel, level + 1);
                }
            }

            return (
                 <div className="setup-screen">
                    {isLastLevel && isLastGrade ? (
                        <>
                            <h1>完全制覇！</h1>
                            <p>全ての学年、全てのレベルをクリアしました。おめでとうございます！</p>
                            <button className="button" onClick={() => setGameState('setup')}>タイトルへ</button>
                        </>
                    ) : (
                        <>
                            <h1>レベル {level} クリア！</h1>
                            <p>正解数: {correctAnswers} / 20</p>
                            <p>{isLastLevel ? `${gradeLevel}年生を制覇！次の学年に進みます。` : '次のレベルに進みます。'}</p>
                            <button className="button" onClick={handleNext}>
                                {isLastLevel ? `小学${gradeLevel + 1}年生に挑戦` : `レベル ${level + 1} へ`}
                            </button>
                        </>
                    )}
                </div>
            )
        } else { // Lost
            return (
                <div className="setup-screen">
                    <h1>挑戦失敗</h1>
                    <p>正解数: {correctAnswers} / 20</p>
                    <p>クリアには18問以上の正解が必要です。もう一度挑戦しますか？</p>
                    <div style={{display: 'flex', gap: '1rem'}}>
                        <button className="button" onClick={() => startLevel(gradeLevel, level)}>再挑戦</button>
                        <button className="button" onClick={() => setGameState('setup')}>学年選択に戻る</button>
                    </div>
                </div>
            );
        }
    }

    return (
        <div className="app-container">
            {gameState === 'setup' && (
                <div className="setup-screen">
                    <h1>漢字クエスト</h1>
                    <h2>AIの冒険</h2>
                    <p>学年を選んで、壮大な冒険に出かけよう。AIが君だけのために物語と世界を創り出す。旅の中で漢字を学び、ハイスコアを目指せ！</p>
                    <select
                        value={gradeLevel}
                        onChange={(e) => setGradeLevel(Number(e.target.value))}
                        aria-label="学年を選択"
                    >
                        {[1, 2, 3, 4, 5, 6].map(g => (
                            <option key={g} value={g} disabled={g > maxGradeUnlocked}>
                                小学{g}年生 {g > maxGradeUnlocked ? '(ロック中)' : ''}
                            </option>
                        ))}
                    </select>
                    <button className="button" onClick={() => startLevel(gradeLevel, 1)}>冒険を始める</button>
                </div>
            )}
            
            {gameState === 'error' && (
                 <div className="setup-screen">
                    <h2>エラーが発生しました</h2>
                    <p>{error}</p>
                    <button className="button" onClick={() => setGameState('setup')}>最初からやり直す</button>
                </div>
            )}

            {gameState === 'level-transition' && renderLevelTransition()}

            {(gameState === 'playing' || gameState === 'feedback') && currentScene && (
                 <div className="game-screen">
                    <div className="game-hud">
                        <span>{gradeLevel}年生 / Lv. {level}</span>
                        <span>{questionNumber} / 20問</span>
                        <span>正解: {correctAnswers}</span>
                    </div>
                    <div className="image-container">
                        {currentImage ? (
                            <img src={currentImage} alt={currentScene.imagePrompt} />
                        ) : (
                            <div className="loader-overlay" style={{position: 'relative', backgroundColor: '#000'}}>
                                <div className="spinner"></div>
                                <p>情景を生成中...</p>
                            </div>
                        )}
                    </div>
                    <div className="story-container">
                        <p>{renderStoryWithHighlight(currentScene.story, currentScene.questionWord)}</p>
                    </div>
                    <div className="interaction-container">
                        <h2>{renderQuestion()}</h2>
                        <form onSubmit={handleSubmit}>
                            <input
                                type="text"
                                value={userAnswer}
                                onChange={(e) => setUserAnswer(e.target.value)}
                                placeholder={currentScene.questionType === 'writing' ? '漢字で入力' : 'ひらがなで入力'}
                                className={feedback || ''}
                                disabled={!!feedback}
                                aria-label="答えを入力"
                                autoFocus
                            />
                            <button type="submit" className="button" disabled={!!feedback || !userAnswer}>答える</button>
                        </form>
                    </div>
                </div>
            )}

            {gameState === 'loading' && (
                 <div className="loader-overlay">
                    <div className="spinner"></div>
                    <p>物語を生成中...</p>
                </div>
            )}
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(<App />);